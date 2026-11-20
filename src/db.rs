use rusqlite::Connection;
use serde_json::Value;
use std::fs;
use std::path::Path;
use uuid::Uuid;

use crate::adapter::filter_narration;
use crate::protobuf::{
    extract_text_from_step_payload, extract_thought_from_step_payload,
    extract_tool_update_from_step_payload, extract_user_text_from_step_payload,
    message_chunk_update,
};

#[cfg(test)]
use crate::types::ConversationDelta;

/// Read the conversation ID created by one specific `agy` invocation.
///
/// The caller gives every invocation its own `--log-file`, so unlike a global
/// directory diff this remains deterministic when several adapters start at
/// the same time.
pub fn find_created_conversation_id_in_log(log_path: &Path) -> Option<String> {
    const MARKER: &str = "Created conversation ";
    let contents = fs::read_to_string(log_path).ok()?;
    contents.lines().find_map(|line| {
        let candidate = line.split_once(MARKER)?.1.split_whitespace().next()?;
        Uuid::parse_str(candidate).ok().map(|id| id.to_string())
    })
}

pub fn agy_run_timed_out(log_path: &Path) -> bool {
    fs::read_to_string(log_path)
        .map(|contents| contents.contains("Print mode: timed out"))
        .unwrap_or(false)
}

pub fn agy_run_permission_denied(log_path: &Path) -> bool {
    fs::read_to_string(log_path)
        .map(|contents| {
            contents.contains("Print mode: soft-denying tool confirmation")
                || contents.contains("permission check failed:")
        })
        .unwrap_or(false)
}

pub fn agy_run_quota_reset(log_path: &Path) -> Option<String> {
    let contents = fs::read_to_string(log_path).ok()?;
    contents.lines().rev().find_map(|line| {
        if !line.contains("Individual quota reached") {
            return None;
        }
        let (_, reset_tail) = line.split_once("Resets in ")?;
        let reset = reset_tail.split('.').next()?.trim();
        if reset.is_empty()
            || reset.len() > 32
            || !reset
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
        {
            return None;
        }
        Some(reset.to_string())
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgyRunFailureKind {
    NativeStreamLag,
    EligibilityNetworkTransient,
}

impl AgyRunFailureKind {
    pub const fn stable_code(self) -> &'static str {
        match self {
            Self::NativeStreamLag => "AGY_NATIVE_STREAM_LAG",
            Self::EligibilityNetworkTransient => "AGY_ELIGIBILITY_NETWORK_TRANSIENT",
        }
    }

    pub const fn message(self) -> &'static str {
        match self {
            Self::NativeStreamLag => {
                "Agy native stream lagged behind; the turn ended without a trustworthy handback"
            }
            Self::EligibilityNetworkTransient => {
                "Agy could not complete its account eligibility check because the Google connection or service was temporarily unavailable"
            }
        }
    }

    pub const fn recovery_state(self) -> &'static str {
        match self {
            Self::NativeStreamLag => "side_effects_possible_no_replay",
            Self::EligibilityNetworkTransient => "prompt_not_started_retry_safe",
        }
    }

    pub const fn recovery_instruction(self) -> &'static str {
        match self {
            Self::NativeStreamLag => {
                "Prompt may already have caused side effects; do not replay it. Rotate to a fresh Agy conversation/seat with bounded context handoff; preserve the old conversation and run log."
            }
            Self::EligibilityNetworkTransient => {
                "The prompt was not started. Retry up to two times with a short delay; re-authenticate only if the failure persists."
            }
        }
    }

    pub const fn safe_to_retry_before_output(self) -> bool {
        matches!(self, Self::EligibilityNetworkTransient)
    }
}

pub fn classify_agy_run_failure(log_path: &Path) -> Option<AgyRunFailureKind> {
    let contents = fs::read_to_string(log_path).ok()?;
    let has_slow_publish = contents.lines().any(|line| {
        line.contains("pubsub: Publish killing slow subscriber")
            && line.contains("subscriber fell behind updates, stalled for ")
    });
    let has_agent_state_stream_error = contents.lines().any(|line| {
        line.contains("StreamAgentStateUpdates error")
            && line.contains("subscriber fell behind updates, stalled for ")
    });
    let has_conversation_stream_error = contents.lines().any(|line| {
        line.contains("Conversation update stream")
            && line.contains("failed: subscriber fell behind updates, stalled for ")
    });
    let has_print_mode_stream_error = contents.lines().any(|line| {
        line.contains("Print mode: conversation update stream failed")
            && line.contains("subscriber fell behind updates, stalled for ")
    });
    let has_userinfo_transport_eof = contents.lines().any(|line| {
        (line.contains("https://www.googleapis.com/oauth2/v2/userinfo")
            || line.contains("loadCodeAssist")
            || line.contains("fetchUserInfo"))
            && (line.contains(": EOF")
                || line.contains("context deadline exceeded")
                || line.contains("connection reset"))
    });
    let has_eligibility_failure = contents.lines().any(|line| {
        line.contains("Print mode: eligibility check failed:")
            || line.contains("Not sending user message: Eligibility check failed:")
            || (line.contains("Print mode: SendUserMessage failed:")
                && line.contains("Eligibility check failed:"))
    });
    let has_eligibility_service_unavailable = contents.lines().any(|line| {
        line.contains("UNAVAILABLE")
            && line.contains("503")
            && line
                .to_ascii_lowercase()
                .contains("service is currently unavailable")
    });

    if has_eligibility_failure
        && (has_userinfo_transport_eof || has_eligibility_service_unavailable)
    {
        Some(AgyRunFailureKind::EligibilityNetworkTransient)
    } else if has_slow_publish
        && has_agent_state_stream_error
        && has_conversation_stream_error
        && has_print_mode_stream_error
    {
        Some(AgyRunFailureKind::NativeStreamLag)
    } else {
        None
    }
}

pub fn find_conversation_id_by_pid(_pid: u32, _conversations_dir: &Path) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let fd_dir = format!("/proc/{}/fd", _pid);
        if let Ok(entries) = std::fs::read_dir(fd_dir) {
            let conversations_dir_canonical = _conversations_dir
                .canonicalize()
                .unwrap_or_else(|_| _conversations_dir.to_path_buf());
            for entry in entries.filter_map(|e| e.ok()) {
                if let Ok(target) = std::fs::read_link(entry.path()) {
                    if target.extension().map(|ext| ext == "db").unwrap_or(false) {
                        if let Some(parent) = target.parent() {
                            let target_parent = parent
                                .canonicalize()
                                .unwrap_or_else(|_| parent.to_path_buf());
                            if target_parent == conversations_dir_canonical {
                                if let Some(stem) = target.file_stem() {
                                    return Some(stem.to_string_lossy().to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("lsof")
            .args(&["-p", &_pid.to_string(), "-Fn"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let conversations_dir_canonical = _conversations_dir
                    .canonicalize()
                    .unwrap_or_else(|_| _conversations_dir.to_path_buf());
                for line in stdout.lines() {
                    if line.starts_with('n') {
                        let path_str = &line[1..];
                        let path = Path::new(path_str);
                        if path.extension().map(|ext| ext == "db").unwrap_or(false) {
                            if let Some(parent) = path.parent() {
                                let target_parent = parent
                                    .canonicalize()
                                    .unwrap_or_else(|_| parent.to_path_buf());
                                if target_parent == conversations_dir_canonical {
                                    if let Some(stem) = path.file_stem() {
                                        return Some(stem.to_string_lossy().to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

pub fn read_rows_from_db(
    conversations_dir: &Path,
    conversation_id: &str,
    after_step_idx: i64,
) -> Option<Vec<(i64, i64, Vec<u8>)>> {
    let db_path = conversations_dir.join(format!("{}.db", conversation_id));
    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;

    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='steps'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);
    if !table_exists {
        eprintln!(
            "[agy-acp] WARN: steps table not found in {}.db — schema changed?",
            conversation_id
        );
        return None;
    }

    let mut stmt = conn
        .prepare("SELECT idx, step_type, step_payload FROM steps WHERE idx > ?1 ORDER BY idx")
        .ok()?;
    let rows: Vec<(i64, i64, Vec<u8>)> = stmt
        .query_map([after_step_idx], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .ok()?
        .filter_map(|r| r.ok())
        .collect();
    Some(rows)
}

pub fn read_replay_updates_from_db(
    conversations_dir: &Path,
    conversation_id: &str,
    skip_naration: bool,
) -> Option<(Vec<Value>, i64)> {
    let rows = read_rows_from_db(conversations_dir, conversation_id, -1)?;
    let mut max_idx = -1;
    let mut updates = Vec::new();
    let mut pending_agent_parts = Vec::new();
    let mut pending_thought_parts = Vec::new();

    for (idx, step_type, payload) in &rows {
        max_idx = max_idx.max(*idx);
        if *step_type == 14 {
            flush_agent_message(&mut pending_agent_parts, &mut updates, skip_naration);
            flush_thought_message(&mut pending_thought_parts, &mut updates);
            if let Some(text) = extract_user_text_from_step_payload(payload) {
                updates.push(message_chunk_update("user_message_chunk", text));
            }
        } else if *step_type == 15 {
            if let Some(text) = extract_text_from_step_payload(payload) {
                if !text.is_empty() {
                    pending_agent_parts.push(text);
                }
            }
            if let Some(text) = extract_thought_from_step_payload(payload) {
                pending_thought_parts.push(text);
            }
        } else if let Some(update) =
            extract_tool_update_from_step_payload(*idx, *step_type, payload)
        {
            flush_agent_message(&mut pending_agent_parts, &mut updates, skip_naration);
            flush_thought_message(&mut pending_thought_parts, &mut updates);
            updates.push(update);
        }
    }
    flush_agent_message(&mut pending_agent_parts, &mut updates, skip_naration);
    flush_thought_message(&mut pending_thought_parts, &mut updates);

    if updates.is_empty() {
        return None;
    }
    Some((updates, max_idx))
}

#[cfg(test)]
pub fn read_delta_from_db(
    conversations_dir: &Path,
    conversation_id: &str,
    after_step_idx: i64,
) -> Option<ConversationDelta> {
    let rows = read_rows_from_db(conversations_dir, conversation_id, after_step_idx)?;

    let mut max_idx = after_step_idx;
    let mut response_parts: Vec<String> = Vec::new();
    for (idx, step_type, payload) in &rows {
        max_idx = max_idx.max(*idx);
        if *step_type == 15 {
            if let Some(text) = extract_text_from_step_payload(payload) {
                if !text.is_empty() {
                    response_parts.push(text);
                }
            }
        }
    }
    if response_parts.is_empty() {
        let response_rows: Vec<_> = rows
            .iter()
            .filter(|(_, step_type, _)| *step_type == 15)
            .collect();
        if !response_rows.is_empty() {
            let payload_sizes: Vec<usize> = response_rows.iter().map(|(_, _, p)| p.len()).collect();
            eprintln!(
                "[agy-acp] WARN: {} new response steps found (payload sizes: {:?}) but none had extractable text \
                 (field 20.1 missing — schema change?)",
                response_rows.len(), payload_sizes
            );
        }
        return None;
    }
    let text = if response_parts.is_empty() {
        None
    } else {
        Some(response_parts.join("\n"))
    };
    Some(ConversationDelta {
        text,
        max_step_idx: max_idx,
    })
}

fn flush_agent_message(parts: &mut Vec<String>, updates: &mut Vec<Value>, skip_naration: bool) {
    if parts.is_empty() {
        return;
    }
    let text = if skip_naration {
        filter_narration(parts)
    } else {
        Some(parts.join("\n"))
    };
    parts.clear();
    if let Some(text) = text {
        if !text.is_empty() {
            updates.push(message_chunk_update("agent_message_chunk", text));
        }
    }
}

fn flush_thought_message(parts: &mut Vec<String>, updates: &mut Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    let text = parts.join("\n");
    parts.clear();
    if !text.is_empty() {
        updates.push(message_chunk_update("agent_thought_chunk", text));
    }
}
