use serde_json::json;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::adapter::is_narration;
use crate::db::{find_created_conversation_id_in_log, read_rows_from_db};
use crate::protobuf::{
    extract_text_from_step_payload, extract_thought_from_step_payload,
    extract_title_from_step_payload, extract_tool_update_from_step_payload,
};
use crate::types::{JsonRpcNotification, StreamingState};

pub fn poll_streaming_delta(
    conversations_dir: &Path,
    run_log_path: Option<&Path>,
    session_id: &str,
    state: &Arc<Mutex<StreamingState>>,
) -> Vec<String> {
    let (conversation_id, after_step_idx) = {
        let mut guard = state.lock().unwrap();
        if let Some(log_path) = run_log_path {
            if let Some(created_id) = find_created_conversation_id_in_log(log_path) {
                if guard.conversation_id.as_deref() != Some(created_id.as_str()) {
                    // Agy can reject a stale --conversation value and create a
                    // replacement conversation in the same invocation.  The
                    // invocation-specific log is authoritative in that case;
                    // retaining the stale ID makes the poller read a missing
                    // DB forever and Paseo receives no live updates.
                    guard.conversation_id = Some(created_id);
                    guard.base_step_idx = -1;
                    guard.last_step_idx = -1;
                    guard.had_agent_text = false;
                    guard.agent_text_lengths.clear();
                    guard.thought_text_lengths.clear();
                    guard.tool_call_statuses.clear();
                    guard.last_title = None;
                }
            }
        }
        if guard.conversation_id.is_none() {
            if let Some(pid) = guard.child_pid {
                guard.conversation_id =
                    crate::db::find_conversation_id_by_pid(pid, conversations_dir);
            }
        }
        // Agy may update the current step payload while it streams. Re-read
        // only the active tail row plus new rows; scanning from the turn's
        // base index on every 500 ms poll makes long turns increasingly slow.
        let after_step_idx = guard
            .last_step_idx
            .saturating_sub(1)
            .max(guard.base_step_idx);
        (guard.conversation_id.clone(), after_step_idx)
    };

    let Some(conversation_id) = conversation_id else {
        return Vec::new();
    };

    let Some(rows) = read_rows_from_db(conversations_dir, &conversation_id, after_step_idx) else {
        return Vec::new();
    };

    let mut guard = state.lock().unwrap();
    let mut notifications = Vec::new();

    for (idx, step_type, payload) in rows {
        guard.last_step_idx = guard.last_step_idx.max(idx);

        if step_type == 15 {
            if !guard.suppress_agent_text {
                if let Some(text) = extract_text_from_step_payload(&payload) {
                    let written_len = guard.agent_text_lengths.get(&idx).copied().unwrap_or(0);
                    if text.len() > written_len {
                        if guard.skip_naration && is_narration(&text) {
                            guard.agent_text_lengths.insert(idx, text.len());
                        } else if let Some(new_text) = text.get(written_len..) {
                            guard.agent_text_lengths.insert(idx, text.len());
                            if !new_text.is_empty() {
                                guard.had_agent_text = true;
                                notifications.push(
                                    serde_json::to_string(&JsonRpcNotification {
                                        jsonrpc: "2.0",
                                        method: "session/update".to_string(),
                                        params: json!({
                                            "sessionId": session_id,
                                            "update": {
                                                "sessionUpdate": "agent_message_chunk",
                                                "content": { "type": "text", "text": new_text },
                                            },
                                        }),
                                    })
                                    .unwrap(),
                                );
                            }
                        }
                    }
                }
            }

            if let Some(thought) = extract_thought_from_step_payload(&payload) {
                let written_len = guard.thought_text_lengths.get(&idx).copied().unwrap_or(0);
                if thought.len() > written_len {
                    let Some(new_text) = thought.get(written_len..) else {
                        continue;
                    };
                    guard.thought_text_lengths.insert(idx, thought.len());
                    if !new_text.is_empty() {
                        notifications.push(
                            serde_json::to_string(&JsonRpcNotification {
                                jsonrpc: "2.0",
                                method: "session/update".to_string(),
                                params: json!({
                                    "sessionId": session_id,
                                    "update": {
                                        "sessionUpdate": "agent_thought_chunk",
                                        "content": { "type": "text", "text": new_text },
                                    },
                                }),
                            })
                            .unwrap(),
                        );
                    }
                }
            }
        } else if step_type == 23 {
            let Some(title) = extract_title_from_step_payload(&payload) else {
                continue;
            };
            if guard.last_title.as_deref() == Some(title.as_str()) {
                continue;
            }
            guard.last_title = Some(title.clone());
            notifications.push(
                serde_json::to_string(&JsonRpcNotification {
                    jsonrpc: "2.0",
                    method: "session/update".to_string(),
                    params: json!({
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "session_info_update",
                            "title": title,
                        },
                    }),
                })
                .unwrap(),
            );
        } else {
            if let Some(mut update) =
                extract_tool_update_from_step_payload(idx, step_type, &payload)
            {
                let Some(tool_call_id) = update
                    .get("toolCallId")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                else {
                    continue;
                };
                let status = update
                    .get("status")
                    .and_then(|value| value.as_str())
                    .unwrap_or("in_progress")
                    .to_string();
                if let Some(previous_status) = guard.tool_call_statuses.get(&tool_call_id) {
                    if previous_status == "completed" || previous_status == "failed" {
                        continue;
                    }
                    if status == "in_progress" {
                        continue;
                    }
                    update["sessionUpdate"] = json!("tool_call_update");
                }
                guard.tool_call_statuses.insert(tool_call_id, status);
                notifications.push(
                    serde_json::to_string(&JsonRpcNotification {
                        jsonrpc: "2.0",
                        method: "session/update".to_string(),
                        params: json!({
                            "sessionId": session_id,
                            "update": update,
                        }),
                    })
                    .unwrap(),
                );
            }
        }
    }

    notifications
}

pub fn finalize_pending_tool_calls(
    session_id: &str,
    state: &Arc<Mutex<StreamingState>>,
    status: &str,
) -> Vec<String> {
    let pending_ids = {
        let mut guard = state.lock().unwrap();
        guard
            .tool_call_statuses
            .iter_mut()
            .filter_map(|(tool_call_id, current_status)| {
                if current_status == "in_progress" {
                    *current_status = "failed".to_string();
                    Some(tool_call_id.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };

    pending_ids
        .into_iter()
        .map(|tool_call_id| {
            serde_json::to_string(&JsonRpcNotification {
                jsonrpc: "2.0",
                method: "session/update".to_string(),
                params: json!({
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": tool_call_id,
                        "status": "failed",
                        "rawOutput": {
                            "resultType": "unverified",
                            "isError": true,
                            "turnStatus": status,
                            "summary": "Turn ended without an observed tool receipt; the tool outcome is unverified."
                        },
                    },
                }),
            })
            .unwrap()
        })
        .collect()
}
