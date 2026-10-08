//! One HTTP(S) request over ureq with rustls, for the model hub client.
//!
//! The crate already carries ureq and rustls for fastokens' own hub access, so
//! the engine gets a TLS client with no system library behind it: the same
//! code runs on macOS, Linux and iOS. Everything above the transport (the Hub
//! API, redirects, ranges, the cache layout, verification) lives in C++.

use std::ffi::{c_char, c_void, CString};
use std::io::Read;
use std::sync::OnceLock;
use std::time::Duration;

use crate::{cstr, guard, set_error};

/// Receives the status line and the response headers once, before any body.
/// `headers` is "name: value\n" per header, names lowercased. Nonzero aborts.
pub type HeadFn = unsafe extern "C" fn(ctx: *mut c_void, status: i32, headers: *const c_char) -> i32;
/// Receives the body in order. Nonzero aborts the transfer.
pub type BodyFn = unsafe extern "C" fn(ctx: *mut c_void, data: *const u8, len: usize) -> i32;

const ABORTED: &str = "aborted by the caller";

fn agent(connect_ms: u32, read_ms: u32) -> ureq::Agent {
    // Redirects are the caller's: the hub answers a weights request with a
    // redirect to a CDN, and the headers that identify the file (X-Repo-Commit,
    // X-Linked-Etag, X-Linked-Size) are on that first response, not the last.
    static DEFAULT: OnceLock<ureq::Agent> = OnceLock::new();
    let build = |c: u32, r: u32| {
        ureq::AgentBuilder::new()
            .redirects(0)
            .timeout_connect(Duration::from_millis(c as u64))
            .timeout_read(Duration::from_millis(r as u64))
            .build()
    };
    if connect_ms == 0 && read_ms == 0 {
        return DEFAULT.get_or_init(|| build(30_000, 120_000)).clone();
    }
    build(
        if connect_ms == 0 { 30_000 } else { connect_ms },
        if read_ms == 0 { 120_000 } else { read_ms },
    )
}

fn deliver(resp: ureq::Response, on_head: HeadFn, on_body: BodyFn, ctx: *mut c_void) -> Result<(), String> {
    let status = resp.status() as i32;
    let mut text = String::new();
    for name in resp.headers_names() {
        for value in resp.all(&name) {
            text.push_str(&name);
            text.push_str(": ");
            text.push_str(value);
            text.push('\n');
        }
    }
    let headers = CString::new(text.replace('\0', "")).unwrap_or_default();
    if unsafe { on_head(ctx, status, headers.as_ptr()) } != 0 {
        return Err(ABORTED.into());
    }
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("reading the response body: {e}"))?;
        if n == 0 {
            return Ok(());
        }
        if unsafe { on_body(ctx, buf.as_ptr(), n) } != 0 {
            return Err(ABORTED.into());
        }
    }
}

/// Sends one request and streams the answer through the callbacks. An HTTP
/// error status is an answer, not a failure: it reaches `on_head` like any
/// other. Returns 0 when the whole response was delivered, -2 when a callback
/// aborted it, and -1 on a transport failure with fk_last_error() set.
#[no_mangle]
pub unsafe extern "C" fn fk_http_request(
    method: *const c_char,
    url: *const c_char,
    request_headers: *const c_char,
    connect_timeout_ms: u32,
    read_timeout_ms: u32,
    on_head: Option<HeadFn>,
    on_body: Option<BodyFn>,
    ctx: *mut c_void,
) -> i32 {
    let mut aborted = false;
    let rc = guard(|| {
        let method = cstr(method)?;
        let url = cstr(url)?;
        let (Some(on_head), Some(on_body)) = (on_head, on_body) else {
            return Err("fk_http_request needs both callbacks".into());
        };
        let mut req = agent(connect_timeout_ms, read_timeout_ms).request(method, url);
        if !request_headers.is_null() {
            for line in cstr(request_headers)?.lines() {
                if let Some((name, value)) = line.split_once(':') {
                    req = req.set(name.trim(), value.trim());
                }
            }
        }
        let resp = match req.call() {
            Ok(r) => r,
            Err(ureq::Error::Status(_, r)) => r,
            Err(ureq::Error::Transport(t)) => return Err(format!("{method} {url}: {t}")),
        };
        let out = deliver(resp, on_head, on_body, ctx);
        if let Err(msg) = &out {
            aborted = msg == ABORTED;
        }
        out
    });
    if rc != 0 && aborted {
        set_error(ABORTED);
        return -2;
    }
    rc
}
