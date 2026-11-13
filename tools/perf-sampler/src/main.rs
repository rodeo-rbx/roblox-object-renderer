//! A user-mode sampling profiler for one process (no admin needed for your own processes): every
//! `interval` ms it reads each thread's CPU time and, for threads that ran since the last round,
//! suspends them, walks their stack (dbghelp StackWalk64 with the target's PDBs) and adds the CPU
//! time used to every function on the stack (inclusive) and to the leaf (exclusive).
//!   sampler <pid> <seconds> [interval_ms] [top]
use std::collections::HashMap;
use std::ffi::c_void;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE};
use windows_sys::Win32::System::Diagnostics::Debug::{
	ADDRESS64, AddrModeFlat, CONTEXT, CONTEXT_FULL_AMD64, GetThreadContext, STACKFRAME64, SYMBOL_INFOW, SYMOPT_DEFERRED_LOADS, SYMOPT_UNDNAME, StackWalk64, SymFromAddrW, SymFunctionTableAccess64, SymGetModuleBase64,
	SymInitializeW, SymRefreshModuleList, SymSetOptions,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next};
use windows_sys::Win32::System::SystemInformation::IMAGE_FILE_MACHINE_AMD64;
use windows_sys::Win32::System::Threading::{GetThreadTimes, OpenProcess, OpenThread, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME};

#[repr(C, align(16))]
struct AlignedContext(CONTEXT);

fn filetime(time: FILETIME) -> u64 {
	((time.dwHighDateTime as u64) << 32) | time.dwLowDateTime as u64
}

fn threads(pid: u32) -> Vec<u32> {
	let mut ids = Vec::new();
	unsafe {
		let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
		let mut entry: THREADENTRY32 = std::mem::zeroed();
		entry.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
		if Thread32First(snapshot, &mut entry) != 0 {
			loop {
				if entry.th32OwnerProcessID == pid {
					ids.push(entry.th32ThreadID);
				}
				if Thread32Next(snapshot, &mut entry) == 0 {
					break;
				}
			}
		}
		CloseHandle(snapshot);
	}
	ids
}

fn walk(process: HANDLE, thread: HANDLE, depth: usize) -> Vec<u64> {
	let mut stack = Vec::new();
	unsafe {
		let mut context: AlignedContext = std::mem::zeroed();
		context.0.ContextFlags = CONTEXT_FULL_AMD64;
		if GetThreadContext(thread, &mut context.0) == 0 {
			return stack;
		}
		let mut frame: STACKFRAME64 = std::mem::zeroed();
		frame.AddrPC = ADDRESS64 { Offset: context.0.Rip, Segment: 0, Mode: AddrModeFlat };
		frame.AddrFrame = ADDRESS64 { Offset: context.0.Rbp, Segment: 0, Mode: AddrModeFlat };
		frame.AddrStack = ADDRESS64 { Offset: context.0.Rsp, Segment: 0, Mode: AddrModeFlat };
		for _ in 0..depth {
			let ok = StackWalk64(IMAGE_FILE_MACHINE_AMD64 as u32, process, thread, &mut frame, &mut context.0 as *mut CONTEXT as *mut c_void, None, Some(SymFunctionTableAccess64), Some(SymGetModuleBase64), None);
			if ok == 0 || frame.AddrPC.Offset == 0 {
				break;
			}
			stack.push(frame.AddrPC.Offset);
		}
	}
	stack
}

fn name(process: HANDLE, address: u64, cache: &mut HashMap<u64, String>) -> String {
	if let Some(name) = cache.get(&address) {
		return name.clone();
	}
	let resolved = unsafe {
		let mut buffer = vec![0u8; std::mem::size_of::<SYMBOL_INFOW>() + 1024 * 2];
		let symbol = buffer.as_mut_ptr() as *mut SYMBOL_INFOW;
		(*symbol).SizeOfStruct = std::mem::size_of::<SYMBOL_INFOW>() as u32;
		(*symbol).MaxNameLen = 1000;
		let mut displacement = 0u64;
		if SymFromAddrW(process, address, &mut displacement, symbol) != 0 {
			let length = (*symbol).NameLen as usize;
			let slice = std::slice::from_raw_parts((*symbol).Name.as_ptr(), length);
			String::from_utf16_lossy(slice)
		} else {
			format!("0x{address:x}")
		}
	};
	cache.insert(address, resolved.clone());
	resolved
}

/// A Rust symbol without its hash and generic noise, shortened.
fn tidy(name: &str) -> String {
	let mut name = name.to_string();
	if let Some(index) = name.rfind("::h")
		&& name[index + 3..].chars().all(|c| c.is_ascii_hexdigit())
	{
		name.truncate(index);
	}
	if name.len() > 140 {
		name.truncate(140);
	}
	name
}

fn main() {
	let mut args = std::env::args().skip(1);
	let pid: u32 = args.next().expect("pid").parse().expect("pid");
	let seconds: f64 = args.next().map_or(5.0, |value| value.parse().unwrap_or(5.0));
	let interval: u64 = args.next().map_or(5, |value| value.parse().unwrap_or(5));
	let top: usize = args.next().map_or(40, |value| value.parse().unwrap_or(40));
	unsafe {
		let process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, pid);
		assert!(!process.is_null(), "open process {pid}");
		SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
		// SAMPLER_SYMPATH: where the target's PDBs are (its own directory, a build's target/release)
		let path: Option<Vec<u16>> = std::env::var("SAMPLER_SYMPATH").ok().map(|path| path.encode_utf16().chain(std::iter::once(0)).collect());
		SymInitializeW(process, path.as_ref().map_or(std::ptr::null(), |path| path.as_ptr()), 1);
		let mut handles: HashMap<u32, (HANDLE, u64)> = HashMap::new();
		let mut inclusive: HashMap<String, f64> = HashMap::new();
		let mut exclusive: HashMap<String, f64> = HashMap::new();
		let mut stacks_by_leaf: HashMap<(String, String), f64> = HashMap::new();
		let mut names = HashMap::new();
		let mut total = 0.0;
		let mut parked = 0.0;
		let mut cpu_total = 0.0;
		let start = Instant::now();
		let mut last_scan = Instant::now() - Duration::from_secs(10);
		while start.elapsed().as_secs_f64() < seconds {
			if last_scan.elapsed() > Duration::from_millis(200) {
				last_scan = Instant::now();
				// modules loaded since (the GPU driver, a late DLL): their symbols and unwind tables
				SymRefreshModuleList(process);
				for id in threads(pid) {
					handles.entry(id).or_insert_with(|| {
						let handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, id);
						(handle, u64::MAX)
					});
				}
			}
			for (_, (handle, last)) in handles.iter_mut() {
				if handle.is_null() {
					continue;
				}
				let (mut creation, mut exit, mut kernel, mut user) = (std::mem::zeroed(), std::mem::zeroed(), std::mem::zeroed(), std::mem::zeroed());
				if GetThreadTimes(*handle, &mut creation, &mut exit, &mut kernel, &mut user) == 0 {
					continue;
				}
				let cpu = filetime(kernel) + filetime(user);
				let delta = if *last == u64::MAX { 0 } else { cpu.saturating_sub(*last) };
				*last = cpu;
				if delta == 0 {
					continue;
				}
				cpu_total += delta as f64 / 10_000.0;
				// a thread caught running counts the interval (time-based); one caught parked ran before
				// it parked, somewhere this sample cannot tell
				let weight = interval as f64;
				if SuspendThread(*handle) == u32::MAX {
					continue;
				}
				let stack = walk(process, *handle, 48);
				ResumeThread(*handle);
				if stack.is_empty() {
					continue;
				}
				let frames: Vec<String> = stack.iter().map(|address| tidy(&name(process, *address, &mut names))).collect();
				let leaf = &frames[0];
				if ["WaitFor", "WaitOnAddress", "DelayExecution", "RemoveIoCompletion", "WaitForAlertByThreadId", "SignalAndWait", "NtWaitForMultipleObjects", "SleepEx"].iter().any(|wait| leaf.contains(wait)) {
					parked += weight;
					continue;
				}
				total += weight;
				*exclusive.entry(frames[0].clone()).or_default() += weight;
				let mut seen = std::collections::HashSet::new();
				for frame in &frames {
					if seen.insert(frame.clone()) {
						*inclusive.entry(frame.clone()).or_default() += weight;
					}
				}
				// the leaf with its nearest ror frame
				let caller = frames.iter().find(|frame| frame.starts_with("ror::") || frame.starts_with("ror_")).cloned().unwrap_or_default();
				*stacks_by_leaf.entry((frames[0].clone(), caller)).or_default() += weight;
			}
			std::thread::sleep(Duration::from_millis(interval));
		}
		println!("running samples: {total:.0} ms, caught parked {parked:.0} ms; CPU used {cpu_total:.0} ms over {:.1} s", start.elapsed().as_secs_f64());
		let print = |title: &str, map: &HashMap<String, f64>| {
			let mut rows: Vec<_> = map.iter().collect();
			rows.sort_by(|a, b| b.1.total_cmp(a.1));
			println!("\n{title}");
			for (name, ms) in rows.into_iter().take(top) {
				println!("{:7.0} ms {:5.1}%  {name}", ms, 100.0 * ms / total.max(1e-9));
			}
		};
		print("exclusive (leaf)", &exclusive);
		print("inclusive", &inclusive);
		let mut rows: Vec<_> = stacks_by_leaf.iter().collect();
		rows.sort_by(|a, b| b.1.total_cmp(a.1));
		println!("\nleaf <- nearest ror frame");
		for ((leaf, caller), ms) in rows.into_iter().take(top) {
			println!("{:7.0} ms  {leaf}  <-  {caller}", ms);
		}
	}
}
