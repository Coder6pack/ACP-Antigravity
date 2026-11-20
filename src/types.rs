use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Safe default for sessions created by an ACP caller.  Agy starts in a
/// read-only proposal mode; a Lead-approved worker assignment must explicitly
/// unlock `accept-edits` for a bounded write.
pub const DEFAULT_MODE_ID: &str = "plan";
pub const FULL_ACCESS_MODE_ID: &str = "full-access";

pub fn default_mode_id() -> String {
    DEFAULT_MODE_ID.to_string()
}

pub fn legacy_mode_id() -> String {
    // Missing mode metadata is legacy state, not an authorization grant.
    DEFAULT_MODE_ID.to_string()
}

fn default_caller_context() -> Value {
    Value::Object(Default::default())
}

fn default_mcp_servers() -> Value {
    Value::Array(Vec::new())
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub id: Option<Value>,
    pub method: Option<String>,
    pub params: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: &'static str,
    pub method: String,
    pub params: Value,
}

/// Persisted session→conversation mapping stored in ~/.openab/agy-acp/sessions.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionStore {
    pub sessions: HashMap<String, StoredSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSession {
    pub conversation_id: Option<String>,
    /// Last step idx read from SQLite; used for delta extraction.
    #[serde(default)]
    pub last_step_idx: i64,
    /// Selected model ID for this session.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Selected Agy permission mode. Legacy state defaults to read-only plan;
    /// writable mode requires a Lead-approved worker assignment.
    #[serde(default = "legacy_mode_id")]
    pub mode_id: String,
    /// The ACP request context associated with this session.  This remains
    /// JSON so newer ACP fields survive a restore without being discarded.
    #[serde(default = "default_caller_context")]
    pub caller_context: Value,
    /// MCP servers requested for this session. Secret-bearing definitions are
    /// held only in memory and are not persisted across adapter restarts.
    #[serde(default = "default_mcp_servers")]
    pub mcp_servers: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Session {
    pub conversation_id: Option<String>,
    /// Last step idx read from SQLite.
    pub last_step_idx: i64,
    /// Selected model ID for this session.
    pub model_id: Option<String>,
    /// Selected Agy permission mode for this ACP session.
    pub mode_id: String,
    /// The ACP request context associated with this session.
    pub caller_context: Value,
    /// MCP servers requested for this session.
    pub mcp_servers: Value,
}

#[cfg(test)]
pub struct ConversationDelta {
    pub text: Option<String>,
    pub max_step_idx: i64,
}

#[derive(Debug, Default)]
pub struct StreamingState {
    pub conversation_id: Option<String>,
    pub base_step_idx: i64,
    pub last_step_idx: i64,
    pub had_agent_text: bool,
    pub agent_text_lengths: HashMap<i64, usize>,
    pub thought_text_lengths: HashMap<i64, usize>,
    pub tool_call_statuses: HashMap<String, String>,
    pub last_title: Option<String>,
    pub skip_naration: bool,
    pub child_pid: Option<u32>,
    /// Native Agy stream-json owns assistant text when true; SQLite remains a
    /// compatibility side-channel for title, thought and tool updates only.
    pub suppress_agent_text: bool,
}

#[derive(Debug, Default)]
pub struct NativeStreamState {
    pub conversation_id: Option<String>,
    pub emitted_text: String,
    pub error: Option<String>,
    /// Terminal result error retained separately from fatal transport errors so
    /// the adapter can recover a bounded set of workspace-tool failures.
    pub terminal_error: Option<String>,
    pub partial_warning_emitted: bool,
    pub saw_result: bool,
    pub skip_naration: bool,
}
