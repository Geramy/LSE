//! C ABI over fastokens. See include/fastokens_ffi.h for the contract.

use std::cell::RefCell;
use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;

use fastokens::{DecodeStream, Tokenizer};

thread_local! {
    static LAST_ERROR: RefCell<std::ffi::CString> =
        RefCell::new(std::ffi::CString::new("").unwrap());
}

fn set_error(msg: impl Into<Vec<u8>>) {
    let cleaned: Vec<u8> = msg.into().into_iter().filter(|b| *b != 0).collect();
    let c = std::ffi::CString::new(cleaned).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Runs `f`, turning both `Err` and a panic into -1 plus a recorded message.
/// Nothing unwinds across the ABI boundary.
fn guard<F: FnOnce() -> Result<(), String>>(f: F) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(msg)) => {
            set_error(msg);
            -1
        }
        Err(_) => {
            set_error("panic in fastokens ffi");
            -1
        }
    }
}

#[repr(C)]
pub struct FkIds {
    data: *mut u32,
    len: usize,
}

#[repr(C)]
pub struct FkText {
    data: *mut c_char,
    len: usize,
}

fn ids_out(v: Vec<u32>, out: *mut FkIds) {
    let mut boxed = v.into_boxed_slice();
    let ptr_ = boxed.as_mut_ptr();
    let len = boxed.len();
    std::mem::forget(boxed);
    unsafe {
        (*out).data = ptr_;
        (*out).len = len;
    }
}

fn text_out(s: String, out: *mut FkText) {
    let mut boxed = s.into_bytes().into_boxed_slice();
    let ptr_ = boxed.as_mut_ptr() as *mut c_char;
    let len = boxed.len();
    std::mem::forget(boxed);
    unsafe {
        (*out).data = ptr_;
        (*out).len = len;
    }
}

unsafe fn cstr<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Err("null string argument".into());
    }
    CStr::from_ptr(p)
        .to_str()
        .map_err(|e| format!("argument is not valid UTF-8: {e}"))
}

#[no_mangle]
pub extern "C" fn fk_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn fk_from_file(path: *const c_char) -> *mut Tokenizer {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Tokenizer, String> {
        let p = cstr(path)?;
        Tokenizer::from_file(Path::new(p)).map_err(|e| format!("{e}"))
    }));
    match result {
        Ok(Ok(t)) => Box::into_raw(Box::new(t)),
        Ok(Err(msg)) => {
            set_error(msg);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic in fk_from_file");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn fk_from_model(repo_id: *const c_char) -> *mut Tokenizer {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Tokenizer, String> {
        let p = cstr(repo_id)?;
        Tokenizer::from_model(p).map_err(|e| format!("{e}"))
    }));
    match result {
        Ok(Ok(t)) => Box::into_raw(Box::new(t)),
        Ok(Err(msg)) => {
            set_error(msg);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic in fk_from_model");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn fk_free(tok: *mut Tokenizer) {
    if !tok.is_null() {
        drop(Box::from_raw(tok));
    }
}

#[no_mangle]
pub unsafe extern "C" fn fk_encode(
    tok: *const Tokenizer,
    text: *const c_char,
    text_len: usize,
    add_special_tokens: bool,
    out: *mut FkIds,
) -> i32 {
    guard(|| {
        if tok.is_null() || out.is_null() {
            return Err("null tokenizer or output".into());
        }
        let bytes = std::slice::from_raw_parts(text as *const u8, text_len);
        let s = std::str::from_utf8(bytes).map_err(|e| format!("input is not UTF-8: {e}"))?;
        let ids = (*tok)
            .encode_with_special_tokens(s, add_special_tokens)
            .map_err(|e| format!("{e}"))?;
        ids_out(ids, out);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fk_decode(
    tok: *const Tokenizer,
    ids: *const u32,
    count: usize,
    skip_special_tokens: bool,
    out: *mut FkText,
) -> i32 {
    guard(|| {
        if tok.is_null() || out.is_null() {
            return Err("null tokenizer or output".into());
        }
        let slice = if count == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(ids, count)
        };
        let s = (*tok)
            .decode(slice, skip_special_tokens)
            .map_err(|e| format!("{e}"))?;
        text_out(s, out);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fk_vocab_size(tok: *const Tokenizer) -> usize {
    if tok.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| (*tok).vocab_size())).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn fk_token_to_id(
    tok: *const Tokenizer,
    token: *const c_char,
    token_len: usize,
    out: *mut u32,
) -> i32 {
    guard(|| {
        if tok.is_null() || out.is_null() {
            return Err("null tokenizer or output".into());
        }
        let bytes = std::slice::from_raw_parts(token as *const u8, token_len);
        let s = std::str::from_utf8(bytes).map_err(|e| format!("token is not UTF-8: {e}"))?;
        match (*tok).token_to_id(s) {
            Some(id) => {
                *out = id;
                Ok(())
            }
            None => Err(format!("no such token: {s:?}")),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn fk_id_to_token(
    tok: *const Tokenizer,
    id: u32,
    out: *mut FkText,
) -> i32 {
    guard(|| {
        if tok.is_null() || out.is_null() {
            return Err("null tokenizer or output".into());
        }
        match (*tok).id_to_token(id) {
            Some(s) => {
                text_out(s.to_string(), out);
                Ok(())
            }
            None => Err(format!("id {id} is outside the vocabulary")),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn fk_is_special(tok: *const Tokenizer, id: u32) -> bool {
    if tok.is_null() {
        return false;
    }
    catch_unwind(AssertUnwindSafe(|| (*tok).is_special_token(id))).unwrap_or(false)
}

#[no_mangle]
pub extern "C" fn fk_stream_new(skip_special_tokens: bool) -> *mut DecodeStream {
    Box::into_raw(Box::new(DecodeStream::new(Vec::new(), skip_special_tokens)))
}

#[no_mangle]
pub unsafe extern "C" fn fk_stream_step(
    stream: *mut DecodeStream,
    tok: *const Tokenizer,
    ids: *const u32,
    count: usize,
    out: *mut FkText,
) -> i32 {
    guard(|| {
        if stream.is_null() || tok.is_null() || out.is_null() {
            return Err("null stream, tokenizer or output".into());
        }
        let slice = if count == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(ids, count).to_vec()
        };
        match (*stream).step(&*tok, slice) {
            // None means the decoder is holding back an incomplete character;
            // an empty chunk tells the caller to feed more tokens.
            Ok(None) => {
                (*out).data = ptr::null_mut();
                (*out).len = 0;
                Ok(())
            }
            Ok(Some(chunk)) => {
                text_out(chunk, out);
                Ok(())
            }
            Err(msg) => Err(msg),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn fk_stream_free(stream: *mut DecodeStream) {
    if !stream.is_null() {
        drop(Box::from_raw(stream));
    }
}

#[no_mangle]
pub unsafe extern "C" fn fk_ids_free(ids: *mut FkIds) {
    if ids.is_null() || (*ids).data.is_null() {
        return;
    }
    drop(Vec::from_raw_parts((*ids).data, (*ids).len, (*ids).len));
    (*ids).data = ptr::null_mut();
    (*ids).len = 0;
}

#[no_mangle]
pub unsafe extern "C" fn fk_text_free(text: *mut FkText) {
    if text.is_null() || (*text).data.is_null() {
        return;
    }
    drop(Vec::from_raw_parts(
        (*text).data as *mut u8,
        (*text).len,
        (*text).len,
    ));
    (*text).data = ptr::null_mut();
    (*text).len = 0;
}
