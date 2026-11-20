// Only an opaque fingerprint is retained. Credential bytes never enter logs or disk.
#[cfg(windows)]
pub(crate) fn active_credential_fingerprint() -> Option<u64> {
    use std::hash::{Hash, Hasher};

    #[repr(C)]
    struct Credential {
        flags: u32,
        kind: u32,
        target_name: *mut u16,
        comment: *mut u16,
        last_written: [u32; 2],
        blob_size: u32,
        blob: *mut u8,
        persist: u32,
        attribute_count: u32,
        attributes: *mut std::ffi::c_void,
        target_alias: *mut u16,
        user_name: *mut u16,
    }

    #[link(name = "Advapi32")]
    unsafe extern "system" {
        fn CredReadW(
            target: *const u16,
            kind: u32,
            flags: u32,
            credential: *mut *mut Credential,
        ) -> i32;
        fn CredFree(buffer: *mut std::ffi::c_void);
    }

    const CRED_TYPE_GENERIC: u32 = 1;
    let target: Vec<u16> = "gemini:antigravity".encode_utf16().chain(Some(0)).collect();
    let mut credential = std::ptr::null_mut();
    // SAFETY: CredReadW allocates the returned credential; CredFree releases it below.
    if unsafe { CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut credential) } == 0 {
        return None;
    }
    let fingerprint = if credential.is_null() {
        None
    } else {
        // SAFETY: CredReadW returned a valid CREDENTIALW until CredFree is called.
        let record = unsafe { &*credential };
        if record.blob.is_null() || record.blob_size == 0 {
            None
        } else {
            let bytes =
                unsafe { std::slice::from_raw_parts(record.blob, record.blob_size as usize) };
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut hasher);
            Some(hasher.finish())
        }
    };
    unsafe { CredFree(credential.cast()) };
    fingerprint
}

#[cfg(not(windows))]
pub(crate) fn active_credential_fingerprint() -> Option<u64> {
    None
}
