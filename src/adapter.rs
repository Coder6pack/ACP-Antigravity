use base64::Engine;
use fs2::FileExt;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use uuid::Uuid;

#[cfg(test)]
use crate::db::read_delta_from_db;
use crate::db::read_replay_updates_from_db;
use crate::streaming::{finalize_pending_tool_calls, poll_streaming_delta};
use crate::types::*;

// Agy's native default is five minutes, which is too short for long-running
// worker turns. Keep a one-hour bridge default; operators can override it with
// `AGY_PRINT_TIMEOUT`.
pub(crate) const DEFAULT_PRINT_TIMEOUT: &str = "1h";
const MCP_ATTACHMENT_ERROR_CODE: i64 = -32001;
const MCP_ATTACHMENT_UNAVAILABLE: &str =
    "required mcpServers could not be staged securely for this Agy session";
const AGY_WORKER_ERROR_CODE: i64 = -32002;
const AGY_WORKER_STATE_KEY: &str = "_agyWorker";
const AGY_ASSIGNMENT_START: &str = "<agy-assignment-envelope>";
const AGY_ASSIGNMENT_END: &str = "</agy-assignment-envelope>";
const AGY_APPROVAL_MESSAGES: &[&str] = &["IMPLEMENTATION_APPROVED", "REMEDIATION_APPROVED"];
const AGY_IMPLEMENTATION_APPROVAL: &str = "IMPLEMENTATION_APPROVED";
const AGY_REMEDIATION_APPROVAL: &str = "REMEDIATION_APPROVED";
const AGY_MAX_REMEDIATION_ROUND: u64 = 1;
const AGY_WORKER_AGENT: &str = "agy-worker";
const AGY_WORKER_PROFILE: &str = "bounded-worker-v1";

const AGY_WORKER_BLOCKED: &str =
    "Agy is a bounded worker. A valid Lead assignment is required; writable work additionally requires a Lead approval envelope. Return a proposal or evidence handoff to Lead and do not expand scope.";

#[derive(Debug)]
enum SessionRequestError {
    Invalid(String),
    McpUnavailable,
}

#[derive(Clone)]
pub struct Adapter {
    pub sessions: HashMap<String, Session>,
    pub working_dir: String,
    pub conversations_dir: PathBuf,
    pub state_file: PathBuf,
    pub available_models: Vec<String>,
    pub(crate) model_cache: ModelCache,
    pub skip_naration: bool,
    /// Trusted process-launch capability. Session parameters cannot grant it.
    pub allow_full_access: bool,
    /// Legacy compatibility field. Delegated Agy workers never use this value
    /// to bypass permissions; only the separately gated autonomous-root launch
    /// may pass Agy's permission-bypass flag.
    pub auto_approve: bool,
    /// When true or AGY_ALLOW_MCP=1, permits non-empty mcpServers in session requests.
    pub allow_mcp: bool,
    pub(crate) mcp_registry: crate::mcp::McpRegistry,
}

#[derive(Clone, Default)]
pub(crate) struct ModelCache {
    account: Option<u64>,
    by_account: HashMap<u64, Vec<String>>,
}

impl ModelCache {
    pub(crate) fn update(&mut self, account: Option<u64>, discovered: Vec<String>) -> Vec<String> {
        self.account = account;
        if !discovered.is_empty() {
            if let Some(account) = account {
                self.by_account.insert(account, discovered.clone());
            }
            discovered
        } else {
            account
                .and_then(|key| self.by_account.get(&key).cloned())
                .unwrap_or_default()
        }
    }

    #[cfg(test)]
    pub(crate) fn current() -> Self {
        Self {
            account: crate::account::active_credential_fingerprint(),
            ..Default::default()
        }
    }
}

impl Adapter {
    pub const MODEL_CONFIG_ID: &'static str = "model";
    pub const MODE_CONFIG_ID: &'static str = "mode";
    pub const ACCEPT_EDITS_MODE_ID: &'static str = "accept-edits";
    pub const PLAN_MODE_ID: &'static str = "plan";
    pub const FULL_ACCESS_MODE_ID: &'static str = crate::types::FULL_ACCESS_MODE_ID;

    fn is_supported_mode(&self, mode_id: &str) -> bool {
        matches!(mode_id, Self::ACCEPT_EDITS_MODE_ID | Self::PLAN_MODE_ID)
            || (self.allow_full_access && mode_id == crate::types::FULL_ACCESS_MODE_ID)
    }

    fn mcp_servers_are_nonempty(value: &Value) -> bool {
        match value {
            Value::Null => false,
            Value::Array(values) => !values.is_empty(),
            Value::Object(values) => !values.is_empty(),
            _ => true,
        }
    }

    fn mcp_servers_value_is_valid(value: &Value) -> bool {
        matches!(value, Value::Null | Value::Array(_) | Value::Object(_))
    }

    fn requested_mode_from_params(
        &self,
        params: &Value,
        allow_legacy_full_access: bool,
    ) -> Result<Option<String>, String> {
        let object = params
            .as_object()
            .ok_or_else(|| "session params must be an object".to_string())?;
        let direct_mode = object
            .get("modeId")
            .or_else(|| object.get("mode"))
            .or_else(|| object.get("permissionMode"));
        let mode_value = direct_mode.or_else(|| {
            object
                .get("modes")
                .and_then(|modes| modes.get("currentModeId"))
        });

        let Some(mode_value) = mode_value else {
            return Ok(None);
        };
        let mode_id = mode_value
            .as_str()
            .ok_or_else(|| "session mode must be a string".to_string())?;
        // `full-access` was persisted by older Agy/Paseo sessions, but a
        // delegated worker process has no trusted capability for it. Keep
        // restore/resume of those sessions loadable and read-only instead of
        // making history hydration fail with an unsupported-mode error. A new
        // session or an explicit mode mutation still fails closed.
        if allow_legacy_full_access
            && mode_id == Self::FULL_ACCESS_MODE_ID
            && !self.allow_full_access
        {
            return Ok(Some(Self::PLAN_MODE_ID.to_string()));
        }
        if !self.is_supported_mode(mode_id) {
            return Err(format!("unknown mode: {mode_id}"));
        }
        Ok(Some(mode_id.to_string()))
    }

    fn parse_session_params(
        &self,
        params: &Value,
        allow_legacy_full_access: bool,
    ) -> Result<(Value, Option<String>), SessionRequestError> {
        if params.as_object().is_none() {
            return Err(SessionRequestError::Invalid(
                "session params must be an object".to_string(),
            ));
        }

        let mcp_servers = params
            .get("mcpServers")
            .cloned()
            .unwrap_or_else(|| json!([]));
        if !Self::mcp_servers_value_is_valid(&mcp_servers) {
            return Err(SessionRequestError::McpUnavailable);
        }
        if !self.allow_mcp && Self::mcp_servers_are_nonempty(&mcp_servers) {
            return Err(SessionRequestError::McpUnavailable);
        }
        if self.allow_mcp && crate::mcp::normalize_mcp_servers(&mcp_servers).is_err() {
            return Err(SessionRequestError::McpUnavailable);
        }

        let mode_id = self
            .requested_mode_from_params(params, allow_legacy_full_access)
            .map_err(SessionRequestError::Invalid)?;
        Ok((mcp_servers, mode_id))
    }

    fn session_has_required_mcp(session: &Session, allow_mcp: bool) -> bool {
        if allow_mcp {
            return false;
        }
        Self::mcp_servers_are_nonempty(&session.mcp_servers)
            || session
                .caller_context
                .get("mcpServers")
                .map(Self::mcp_servers_are_nonempty)
                .unwrap_or(false)
    }

    fn stored_session_has_required_mcp(session: &StoredSession, allow_mcp: bool) -> bool {
        if allow_mcp {
            return false;
        }
        Self::mcp_servers_are_nonempty(&session.mcp_servers)
            || session
                .caller_context
                .get("mcpServers")
                .map(Self::mcp_servers_are_nonempty)
                .unwrap_or(false)
    }

    fn stored_session_assignment_is_approved(session: &StoredSession) -> bool {
        Self::assignment_from_value(&session.caller_context)
            .map(|assignment| Self::assignment_is_approved(&assignment))
            .unwrap_or(false)
    }

    fn merge_context(base: &mut Value, incoming: &Value) {
        let Value::Object(incoming) = incoming else {
            return;
        };
        let Value::Object(base) = base else {
            let mut filtered = Map::new();
            for (key, value) in incoming {
                if key != "sessionId" && !Self::is_assignment_key(key) {
                    filtered.insert(key.clone(), value.clone());
                }
            }
            *base = Value::Object(filtered);
            return;
        };
        for (key, value) in incoming {
            if key != "sessionId" && !Self::is_assignment_key(key) {
                if let Some(existing) = base.get_mut(key) {
                    if existing.is_object() && value.is_object() {
                        Self::merge_context(existing, value);
                    } else {
                        *existing = value.clone();
                    }
                } else {
                    base.insert(key.clone(), value.clone());
                }
            }
        }
    }

    fn is_assignment_key(key: &str) -> bool {
        matches!(
            key,
            AGY_WORKER_STATE_KEY
                | "agyAssignment"
                | "agy_assignment"
                | "leadAssignment"
                | "lead_assignment"
                | "assignment"
                | "workerAssignment"
        )
    }

    fn nonempty_string(value: Option<&Value>) -> bool {
        value
            .and_then(Value::as_str)
            .is_some_and(|text| !text.trim().is_empty())
    }

    fn nonempty_value(value: Option<&Value>) -> bool {
        match value {
            Some(Value::String(text)) => !text.trim().is_empty(),
            Some(Value::Array(values)) => !values.is_empty(),
            Some(Value::Object(values)) => !values.is_empty(),
            _ => false,
        }
    }

    fn assignment_core_equal(left: &Value, right: &Value) -> bool {
        let Some(left) = left.as_object() else {
            return false;
        };
        let Some(right) = right.as_object() else {
            return false;
        };
        for key in [
            "taskId",
            "parentAgentId",
            "parentRole",
            "objective",
            "modelId",
            "modelProvider",
            "thinkingOptionId",
            "scope",
            "exclusions",
            "verification",
        ] {
            let Some(right_val) = right.get(key) else {
                // Follow-up approval/remediation envelopes may omit unchanged fields
                // already established by the initial Lead assignment for the same taskId.
                continue;
            };
            let Some(left_val) = left.get(key) else {
                continue;
            };
            if key == "modelId" {
                let left_model = left_val.as_str().map(Self::resolve_static_model_alias);
                let right_model = right_val.as_str().map(Self::resolve_static_model_alias);
                if left_model != right_model {
                    return false;
                }
                continue;
            }
            if key == "parentRole" {
                let left_role = left_val.as_str().map(|s| s.to_ascii_lowercase());
                let right_role = right_val.as_str().map(|s| s.to_ascii_lowercase());
                if left_role != right_role {
                    return false;
                }
                continue;
            }
            if left_val != right_val {
                return false;
            }
        }
        true
    }

    fn assignment_approval_is_valid(object: &Map<String, Value>) -> bool {
        let round = object
            .get("remediationRound")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if round > AGY_MAX_REMEDIATION_ROUND {
            return false;
        }
        match object.get("approval").and_then(Value::as_str) {
            None => round == 0,
            Some(AGY_IMPLEMENTATION_APPROVAL) => round == 0,
            Some(AGY_REMEDIATION_APPROVAL) => {
                round == AGY_MAX_REMEDIATION_ROUND
                    && Self::nonempty_value(object.get("acceptedFindings"))
            }
            _ => false,
        }
    }

    /// Agy cannot authenticate the ACP caller itself, so the adapter accepts
    /// only the narrow, explicit Lead envelope and fails closed to plan mode
    /// for everything else.  The envelope is data, not an authority upgrade;
    /// the parent/Lead owns its contents and the resulting candidate.
    fn is_valid_worker_assignment(value: &Value) -> bool {
        let Some(object) = value.as_object() else {
            return false;
        };
        if !Self::nonempty_string(object.get("taskId")) {
            return false;
        }
        if object
            .get("parentRole")
            .and_then(Value::as_str)
            .map(|role| !role.eq_ignore_ascii_case("lead"))
            .unwrap_or(true)
        {
            return false;
        }
        if !Self::assignment_approval_is_valid(object) {
            return false;
        }
        for optional_str in [
            "parentAgentId",
            "modelId",
            "modelProvider",
            "thinkingOptionId",
        ] {
            if let Some(val) = object.get(optional_str) {
                if !val.is_null() && !Self::nonempty_string(Some(val)) {
                    return false;
                }
            }
        }
        if let Some(exclusions) = object.get("exclusions") {
            if !matches!(
                exclusions,
                Value::Null | Value::String(_) | Value::Array(_) | Value::Object(_)
            ) {
                return false;
            }
        }
        let is_followup_approval = object
            .get("approval")
            .and_then(Value::as_str)
            .map(|approval| AGY_APPROVAL_MESSAGES.contains(&approval))
            .unwrap_or(false)
            && !object.contains_key("objective")
            && !object.contains_key("scope")
            && !object.contains_key("verification");
        if is_followup_approval {
            return true;
        }
        if !Self::nonempty_string(object.get("objective")) {
            return false;
        }
        for key in ["scope", "verification"] {
            let Some(value) = object.get(key) else {
                return false;
            };
            let valid = match value {
                Value::String(text) => !text.trim().is_empty(),
                Value::Array(values) => !values.is_empty(),
                Value::Object(values) => !values.is_empty(),
                _ => false,
            };
            if !valid {
                return false;
            }
        }
        true
    }

    fn assignment_from_value(value: &Value) -> Option<Value> {
        if Self::is_valid_worker_assignment(value) {
            return Some(value.clone());
        }
        let object = value.as_object()?;
        for key in [
            AGY_WORKER_STATE_KEY,
            "agyAssignment",
            "agy_assignment",
            "leadAssignment",
            "lead_assignment",
            "assignment",
            "workerAssignment",
        ] {
            if let Some(candidate) = object.get(key) {
                if Self::is_valid_worker_assignment(candidate) {
                    return Some(candidate.clone());
                }
            }
        }
        for key in ["callerContext", "caller_context"] {
            if let Some(context) = object.get(key) {
                if let Some(candidate) = Self::assignment_from_value(context) {
                    return Some(candidate);
                }
            }
        }
        None
    }

    fn assignment_key_present(value: &Value) -> bool {
        let Some(object) = value.as_object() else {
            return false;
        };
        [
            AGY_WORKER_STATE_KEY,
            "agyAssignment",
            "agy_assignment",
            "leadAssignment",
            "lead_assignment",
            "assignment",
            "workerAssignment",
        ]
        .iter()
        .any(|key| object.contains_key(*key))
            || ["callerContext", "caller_context"]
                .iter()
                .filter_map(|key| object.get(*key))
                .any(Self::assignment_key_present)
    }

    fn parse_worker_assignment(params: &Value) -> Result<Option<Value>, String> {
        if !Self::assignment_key_present(params) {
            return Ok(Self::assignment_from_value(params));
        }
        Self::assignment_from_value(params).ok_or_else(|| {
            "invalid Agy assignment: Lead envelope must include taskId, parentAgentId, parentRole=lead, objective, modelId, modelProvider, thinkingOptionId, scope, exclusions, and verification".to_string()
        }).map(Some)
    }

    fn prompt_text_from_params(params: &Value) -> String {
        params
            .get("prompt")
            .and_then(Value::as_array)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    }

    fn assignment_from_prompt(prompt: &str) -> Result<Option<Value>, String> {
        let Some(start) = prompt.find(AGY_ASSIGNMENT_START) else {
            return Ok(None);
        };
        let json_start = start + AGY_ASSIGNMENT_START.len();
        let Some(relative_end) = prompt[json_start..].find(AGY_ASSIGNMENT_END) else {
            return Err("Agy assignment envelope is missing its closing tag".to_string());
        };
        let json_end = json_start + relative_end;
        let assignment: Value = serde_json::from_str(prompt[json_start..json_end].trim())
            .map_err(|error| format!("Agy assignment envelope is invalid JSON: {error}"))?;
        if !Self::is_valid_worker_assignment(&assignment) {
            return Err("Agy assignment envelope failed Lead-authority validation".to_string());
        }
        Ok(Some(assignment))
    }

    fn assignment_from_request(params: &Value) -> Result<Option<Value>, String> {
        let structured = Self::parse_worker_assignment(params)?;
        if structured.is_some() {
            return Ok(structured);
        }
        Self::assignment_from_prompt(&Self::prompt_text_from_params(params))
    }

    fn approval_marker(prompt: &str) -> Option<&'static str> {
        AGY_APPROVAL_MESSAGES.iter().copied().find(|marker| {
            prompt.lines().any(|line| {
                let line = line.trim().trim_matches('`').trim();
                line.eq_ignore_ascii_case(marker)
                    || line.eq_ignore_ascii_case(&format!("<agy-approval>{marker}</agy-approval>"))
                    || line.eq_ignore_ascii_case(&format!("approval: {marker}"))
            })
        })
    }

    fn assignment_is_approved(assignment: &Value) -> bool {
        assignment
            .get("approval")
            .and_then(Value::as_str)
            .map(|approval| AGY_APPROVAL_MESSAGES.contains(&approval))
            .unwrap_or(false)
    }

    fn assignment_with_approval(assignment: &Value, approval: &str) -> Option<Value> {
        if !AGY_APPROVAL_MESSAGES.contains(&approval) {
            return None;
        }
        let mut object = assignment.as_object()?.clone();
        object.insert("approval".to_string(), Value::String(approval.to_string()));
        let assignment = Value::Object(object);
        Self::is_valid_worker_assignment(&assignment).then_some(assignment)
    }

    fn remember_worker_assignment(
        &mut self,
        session_id: &str,
        assignment: &Value,
    ) -> Result<(), String> {
        let existing = self
            .sessions
            .get(session_id)
            .and_then(Self::session_assignment);
        let existing_task = existing.as_ref().and_then(Self::assignment_task_id);
        let incoming_task = Self::assignment_task_id(assignment);
        if existing_task.is_some() && existing_task != incoming_task {
            return Err(format!(
                "Agy session is already bound to Lead task {}; start a new session for task {}",
                existing_task.unwrap_or("unknown"),
                incoming_task.unwrap_or("unknown")
            ));
        }

        let existing_approval = existing
            .as_ref()
            .and_then(|value| value.get("approval"))
            .and_then(Value::as_str);
        let incoming_approval = assignment.get("approval").and_then(Value::as_str);
        if let Some(existing) = existing.as_ref() {
            if Self::assignment_is_approved(existing)
                && !Self::assignment_core_equal(existing, assignment)
            {
                return Err(
                    "approved Agy assignment is immutable; start a new Lead task for a changed scope or model"
                        .to_string(),
                );
            }
        }
        if existing_approval == Some(AGY_REMEDIATION_APPROVAL)
            && incoming_approval != Some(AGY_REMEDIATION_APPROVAL)
        {
            return Err(
                "Agy remediation round is already consumed; remaining findings must become a new Lead task"
                    .to_string(),
            );
        }
        if existing_approval == Some(AGY_REMEDIATION_APPROVAL)
            && incoming_approval == Some(AGY_REMEDIATION_APPROVAL)
            && existing.as_ref() != Some(assignment)
        {
            return Err(
                "Agy remediation approval is already consumed; do not issue a second remediation envelope"
                    .to_string(),
            );
        }

        let mut assignment_to_store = assignment.clone();
        if let (Some(existing_obj), Some(store_obj)) = (
            existing.as_ref().and_then(Value::as_object),
            assignment_to_store.as_object_mut(),
        ) {
            for (key, val) in existing_obj {
                if !store_obj.contains_key(key) {
                    store_obj.insert(key.clone(), val.clone());
                }
            }
        }
        if existing_approval.is_some() && !Self::assignment_is_approved(&assignment_to_store) {
            if let (Some(existing_approval), Some(object)) =
                (existing_approval, assignment_to_store.as_object_mut())
            {
                object.insert(
                    "approval".to_string(),
                    Value::String(existing_approval.to_string()),
                );
            }
        }

        let assigned_model = assignment_to_store
            .get("modelId")
            .and_then(Value::as_str)
            .map(|m| self.resolve_session_model_id(m));
        let existing_resolved_model = self
            .sessions
            .get(session_id)
            .and_then(|s| s.model_id.as_deref())
            .map(|m| self.resolve_session_model_id(m));

        let Some(session) = self.sessions.get_mut(session_id) else {
            return Ok(());
        };
        let before_context = session.caller_context.clone();
        let before_model = session.model_id.clone();
        let before_mode = session.mode_id.clone();
        if let Some(assigned_model) = assigned_model {
            if let Some(existing_model) = existing_resolved_model {
                if existing_model != assigned_model {
                    if existing
                        .as_ref()
                        .map(Self::assignment_is_approved)
                        .unwrap_or(false)
                    {
                        return Err(format!(
                            "Lead assignment model '{}' does not match the approved session model '{}'",
                            assigned_model,
                            existing_model
                        ));
                    }
                    // A proposal may be corrected before approval. Once the
                    // Lead approves it, model selection is immutable.
                    session.model_id = Some(assigned_model);
                }
            } else {
                session.model_id = Some(assigned_model);
            }
        }
        if !session.caller_context.is_object() {
            session.caller_context = Value::Object(Map::new());
        }
        Self::set_worker_assignment(&mut session.caller_context, &assignment_to_store);
        let approval_changed =
            incoming_approval.is_some() && existing_approval != incoming_approval;
        if approval_changed && Self::assignment_is_approved(&assignment_to_store) {
            // A Lead approval is the only transition that can unlock writes.
            // The caller cannot manufacture this transition with mode flags.
            session.mode_id = Self::ACCEPT_EDITS_MODE_ID.to_string();
        }
        let changed = before_context != session.caller_context
            || before_model != session.model_id
            || before_mode != session.mode_id;
        let conversation_id = session.conversation_id.clone();
        let last_step_idx = session.last_step_idx;
        let model_id = session.model_id.clone();
        // End the mutable borrow before normalizing/persisting through `self`.
        let _ = session;
        self.normalize_worker_mode(session_id);
        if changed {
            let (conversation_id, last_step_idx, model_id) = self
                .sessions
                .get(session_id)
                .map(|session| {
                    (
                        session.conversation_id.clone(),
                        session.last_step_idx,
                        session.model_id.clone(),
                    )
                })
                .unwrap_or((conversation_id, last_step_idx, model_id));
            self.persist_session(
                session_id,
                conversation_id.as_deref(),
                last_step_idx,
                model_id.as_deref(),
            );
        }
        Ok(())
    }

    fn set_worker_assignment(context: &mut Value, assignment: &Value) {
        if let Value::Object(object) = context {
            object.insert(AGY_WORKER_STATE_KEY.to_string(), assignment.clone());
        }
    }

    fn session_assignment(session: &Session) -> Option<Value> {
        Self::assignment_from_value(&session.caller_context)
    }

    fn assignment_task_id(value: &Value) -> Option<&str> {
        value.as_object()?.get("taskId")?.as_str()
    }

    fn worker_write_authorized_for_session(&self, session_id: &str) -> bool {
        let Some(session) = self.sessions.get(session_id) else {
            return false;
        };
        if session.mode_id != Self::ACCEPT_EDITS_MODE_ID {
            return false;
        }
        // A direct Paseo seat has no delegated-worker assignment envelope.
        // In that case the host-selected `accept-edits` mode is the user's
        // permission-bounded choice and must reach the native Agy process.
        // Once any assignment marker is present, the stricter delegated
        // worker contract applies and an approval is mandatory.
        if !Self::assignment_key_present(&session.caller_context) {
            return true;
        }
        let Some(assignment) = Self::session_assignment(session) else {
            return false;
        };
        Self::assignment_is_approved(&assignment)
    }

    fn worker_assignment_approved(&self, session_id: &str) -> bool {
        self.sessions
            .get(session_id)
            .and_then(Self::session_assignment)
            .map(|assignment| Self::assignment_is_approved(&assignment))
            .unwrap_or(false)
    }

    fn effective_agy_mode(&self, session_id: &str) -> String {
        let requested = self
            .sessions
            .get(session_id)
            .map(|session| session.mode_id.as_str())
            .unwrap_or(DEFAULT_MODE_ID);
        if requested == Self::FULL_ACCESS_MODE_ID && self.allow_full_access {
            return Self::ACCEPT_EDITS_MODE_ID.to_string();
        }
        if requested == Self::ACCEPT_EDITS_MODE_ID
            && self.worker_write_authorized_for_session(session_id)
        {
            Self::ACCEPT_EDITS_MODE_ID.to_string()
        } else {
            // Direct and proposal-only Agy sessions are always read-only at
            // the native CLI boundary, regardless of a host-selected mode.
            Self::PLAN_MODE_ID.to_string()
        }
    }

    pub(crate) fn agy_auto_approve_value(&self, session_id: &str) -> &'static str {
        if self.effective_agy_mode(session_id) == Self::ACCEPT_EDITS_MODE_ID {
            "1"
        } else {
            "0"
        }
    }

    fn assignment_prompt_context(&self, session_id: &str) -> Option<Value> {
        self.sessions
            .get(session_id)
            .and_then(Self::session_assignment)
    }

    fn normalize_worker_mode(&mut self, session_id: &str) {
        let authorized = self.worker_write_authorized_for_session(session_id);
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.mode_id == Self::ACCEPT_EDITS_MODE_ID && !authorized {
                session.mode_id = Self::PLAN_MODE_ID.to_string();
            }
        }
    }

    fn extract_assignment_marker(prompt: &str) -> Result<(String, Option<Value>), String> {
        let Some(start) = prompt.find(AGY_ASSIGNMENT_START) else {
            return Ok((prompt.to_string(), None));
        };
        let json_start = start + AGY_ASSIGNMENT_START.len();
        let Some(relative_end) = prompt[json_start..].find(AGY_ASSIGNMENT_END) else {
            return Err("Agy assignment envelope is missing its closing tag".to_string());
        };
        let json_end = json_start + relative_end;
        let assignment: Value = serde_json::from_str(prompt[json_start..json_end].trim())
            .map_err(|error| format!("Agy assignment envelope is invalid JSON: {error}"))?;
        if !Self::is_valid_worker_assignment(&assignment) {
            return Err("Agy assignment envelope failed Lead-authority validation".to_string());
        }
        let mut stripped = String::with_capacity(prompt.len());
        stripped.push_str(prompt[..start].trim_end());
        if !stripped.is_empty()
            && !prompt[json_end + AGY_ASSIGNMENT_END.len()..]
                .trim()
                .is_empty()
        {
            stripped.push_str("\n\n");
        }
        stripped.push_str(prompt[json_end + AGY_ASSIGNMENT_END.len()..].trim_start());
        Ok((stripped, Some(assignment)))
    }

    fn update_session_from_request(
        &mut self,
        session_id: &str,
        params: &Value,
        mcp_servers: &Value,
        mode_id: Option<&str>,
    ) -> Result<bool, SessionRequestError> {
        let allow_mcp = self.allow_mcp;
        let Some(session) = self.sessions.get_mut(session_id) else {
            return Ok(false);
        };
        if Self::session_has_required_mcp(session, allow_mcp) {
            return Err(SessionRequestError::McpUnavailable);
        }

        let before_context = session.caller_context.clone();
        let before_mcp = session.mcp_servers.clone();
        let before_mode = session.mode_id.clone();
        Self::merge_context(&mut session.caller_context, params);
        if params.get("mcpServers").is_some() {
            session.mcp_servers = mcp_servers.clone();
        }
        if let Some(mode_id) = mode_id {
            session.mode_id = mode_id.to_string();
        }

        Ok(before_context != session.caller_context
            || before_mcp != session.mcp_servers
            || before_mode != session.mode_id)
    }

    fn error_response(id: Value, code: i64, message: impl Into<String>) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(json!({"code": code, "message": message.into()})),
        }
    }

    fn session_request_error_response(id: Value, error: SessionRequestError) -> JsonRpcResponse {
        match error {
            SessionRequestError::Invalid(message) => Self::error_response(id, -32602, message),
            SessionRequestError::McpUnavailable => {
                Self::error_response(id, MCP_ATTACHMENT_ERROR_CODE, MCP_ATTACHMENT_UNAVAILABLE)
            }
        }
    }

    pub fn merge_prompt_session_from(&mut self, session_id: &str, completed: &Self) {
        if let Some(session) = completed.sessions.get(session_id) {
            self.sessions
                .insert(session_id.to_string(), session.clone());
        }
    }

    pub fn new() -> Self {
        Self::new_with_capabilities(false, false)
    }

    pub fn new_with_skip_naration(skip_naration: bool) -> Self {
        Self::new_with_capabilities(skip_naration, false)
    }

    pub fn new_with_capabilities(skip_naration: bool, allow_full_access: bool) -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| "/tmp".to_string());
        let state_dir = PathBuf::from(&home).join(".openab/agy-acp");
        // Do not derive a permission bypass from ambient environment state.
        // Paseo providers used to set AGY_AUTO_APPROVE=1 globally, which made
        // every Agy session silently writable.  Only the separately gated
        // autonomous-root capability can ever reach the native bypass path.
        let auto_approve = false;
        let allow_mcp = std::env::var("AGY_ALLOW_MCP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let account = crate::account::active_credential_fingerprint();
        let available_models = Self::fetch_available_models(account.is_none());
        let mut model_cache = ModelCache {
            account,
            ..Default::default()
        };
        if let Some(account) = account {
            if !available_models.is_empty() {
                model_cache
                    .by_account
                    .insert(account, available_models.clone());
            }
        }
        Self {
            sessions: HashMap::new(),
            working_dir: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "/tmp".to_string()),
            conversations_dir: PathBuf::from(&home).join(".gemini/antigravity-cli/conversations"),
            state_file: state_dir.join("sessions.json"),
            available_models,
            model_cache,
            skip_naration,
            allow_full_access,
            auto_approve,
            allow_mcp,
            mcp_registry: crate::mcp::McpRegistry::new(),
        }
    }

    fn attach_mcp_for_session(
        &self,
        session_id: &str,
        mcp_servers: &Value,
    ) -> Result<(), SessionRequestError> {
        let attachment = crate::mcp::normalize_mcp_servers(mcp_servers)
            .map_err(|_| SessionRequestError::McpUnavailable)?;
        let Some(attachment) = attachment else {
            return Ok(());
        };
        let state_dir = self
            .state_file
            .parent()
            .ok_or(SessionRequestError::McpUnavailable)?;
        let agent_profile = std::env::var("AGY_AGENT").ok().and_then(|agent| {
            let agent = agent.trim();
            if agent.is_empty()
                || Path::new(agent).file_name().and_then(|name| name.to_str()) != Some(agent)
            {
                return None;
            }
            self.conversations_dir
                .parent()
                .and_then(Path::parent)
                .map(|gemini_dir| {
                    gemini_dir
                        .join("config")
                        .join("agents")
                        .join(format!("{agent}.md"))
                })
                .filter(|path| path.is_file())
        });
        let source_gemini_dir = self.conversations_dir.parent().and_then(Path::parent);
        self.mcp_registry
            .attach(
                session_id,
                attachment,
                state_dir,
                source_gemini_dir,
                agent_profile.as_deref(),
                Self::is_supervisor_runtime(),
            )
            .map(|_| ())
            .map_err(|_| SessionRequestError::McpUnavailable)
    }

    #[cfg(test)]
    pub(crate) fn mcp_env_for_session(&self, session_id: &str) -> Option<Vec<(String, String)>> {
        self.mcp_registry
            .get(session_id)
            .map(|seat| seat.env_vars())
    }

    pub(crate) fn mcp_args_for_session(&self, session_id: &str) -> Result<Option<Vec<String>>, ()> {
        let Some(seat) = self.mcp_registry.get(session_id) else {
            return Ok(None);
        };
        let app_data_dir = self.conversations_dir.parent().ok_or(())?;
        seat.agy_args(app_data_dir).map(Some)
    }

    pub fn normalize_model_id(val: &str) -> String {
        let trimmed = val.trim();
        if let Some(tab_idx) = trimmed.find('\t') {
            let label = trimmed[tab_idx + 1..].trim();
            if !label.is_empty() {
                return label.to_string();
            }
            return trimmed[..tab_idx].trim().to_string();
        }
        trimmed.to_string()
    }

    pub fn resolve_static_model_alias(val: &str) -> String {
        let normalized = Self::normalize_model_id(val);
        match normalized.to_ascii_lowercase().as_str() {
            "gemini-4-argon-high" => "Gemini 4 Argon (High)".to_string(),
            "gemini-4-argon-medium" => "Gemini 4 Argon (Medium)".to_string(),
            "gemini-4-argon-low" => "Gemini 4 Argon (Low)".to_string(),
            "gemini-3-1-pro-high" | "gemini-3.1-pro-high" => "Gemini 3.1 Pro (High)".to_string(),
            "gemini-3-1-pro-medium" | "gemini-3.1-pro-medium" => {
                "Gemini 3.1 Pro (Medium)".to_string()
            }
            "gemini-3-1-pro-low" | "gemini-3.1-pro-low" => "Gemini 3.1 Pro (Low)".to_string(),
            "claude-opus-5-5-high" | "claude-opus-5.5-high" => "Claude Opus 5.5 (High)".to_string(),
            "claude-opus-5-5-medium" | "claude-opus-5.5-medium" => {
                "Claude Opus 5.5 (Medium)".to_string()
            }
            "claude-opus-5-5-low" | "claude-opus-5.5-low" => "Claude Opus 5.5 (Low)".to_string(),
            "claude-sonnet-5-5-high" | "claude-sonnet-5.5-high" => {
                "Claude Sonnet 5.5 (High)".to_string()
            }
            "claude-sonnet-5-5-medium" | "claude-sonnet-5.5-medium" => {
                "Claude Sonnet 5.5 (Medium)".to_string()
            }
            "claude-sonnet-5-5-low" | "claude-sonnet-5.5-low" => {
                "Claude Sonnet 5.5 (Low)".to_string()
            }
            "claude-sonnet-4-6" | "claude-sonnet-4.6" => "Claude Sonnet 4.6".to_string(),
            "claude-sonnet-4-6-thinking" | "claude-sonnet-4.6-thinking" => {
                "ClaudeSonnet 4.6 (Thinking)".to_string()
            }
            "claude-opus-4-6-thinking" | "claude-opus-4.6-thinking" => {
                "Claude Opus 4.6 (Thinking)".to_string()
            }
            "gpt-oss-120b-medium" => "GPT-OSS 120B (Medium)".to_string(),
            "gemma-4-26b-a4b" => "Gemma 4 26B A4B".to_string(),
            "gemma-4-31b" => "Gemma 4 31B".to_string(),
            "gemini-3-flash" | "gemini-3.0-flash" | "gemini-3-flash-agent" => {
                "Gemini 3 Flash".to_string()
            }
            "gemini-2-5-pro" | "gemini-2.5-pro" => "Gemini 2.5 Pro".to_string(),
            "gemini-2-5-flash" | "gemini-2.5-flash" => "Gemini 2.5 Flash".to_string(),
            "gemini-2-5-flash-lite" | "gemini-2.5-flash-lite" => {
                "Gemini 2.5 Flash Lite".to_string()
            }
            "gemini-3-8-flash-high" | "gemini-3.8-flash-high" => {
                "Gemini 3.8 Flash (High)".to_string()
            }
            "gemini-3-8-flash-medium" | "gemini-3.8-flash-medium" => {
                "Gemini 3.8 Flash (Medium)".to_string()
            }
            "gemini-3-8-flash-low" | "gemini-3.8-flash-low" => "Gemini 3.8 Flash (Low)".to_string(),
            "gemini-3-7-flash-high" | "gemini-3.7-flash-high" => {
                "Gemini 3.7 Flash (High)".to_string()
            }
            "gemini-3-7-flash-low" | "gemini-3.7-flash-low" => "Gemini 3.7 Flash (Low)".to_string(),
            "gemini-3-7-pro-high" | "gemini-3.7-pro-high" => "Gemini 3.7 Pro (High)".to_string(),
            "gemini-3-7-pro-low" | "gemini-3.7-pro-low" => "Gemini 3.7 Pro (Low)".to_string(),
            _ => normalized,
        }
    }

    pub fn resolve_session_model_id(&self, val: &str) -> String {
        let candidate = Self::resolve_static_model_alias(val);
        if self.available_models.iter().any(|m| m == &candidate) {
            return candidate;
        }
        if candidate == "Claude Sonnet 4.6 (Thinking)"
            && self
                .available_models
                .iter()
                .any(|m| m == "ClaudeSonnet 4.6 (Thinking)")
        {
            return "ClaudeSonnet 4.6 (Thinking)".to_string();
        }
        if self
            .available_models
            .iter()
            .any(|m| m == "Gemini 4 Argon (High)")
            && matches!(
                candidate.as_str(),
                "Gemini 3.8 Flash (High)" | "Gemini 3.7 Flash (High)" | "Gemini 3.7 Pro (High)"
            )
        {
            return "Gemini 4 Argon (High)".to_string();
        }
        if self
            .available_models
            .iter()
            .any(|m| m == "Gemini 4 Argon (Low)")
            && matches!(
                candidate.as_str(),
                "Gemini 3.8 Flash (Low)" | "Gemini 3.7 Flash (Low)" | "Gemini 3.7 Pro (Low)"
            )
        {
            return "Gemini 4 Argon (Low)".to_string();
        }
        if let Some(found) = self
            .available_models
            .iter()
            .find(|m| m.eq_ignore_ascii_case(&candidate))
        {
            return found.clone();
        }
        candidate
    }

    pub(crate) fn parse_materialized_models(raw: &str) -> Option<Vec<String>> {
        let models: Vec<String> = serde_json::from_str(raw).ok()?;
        let models: Vec<String> = models
            .into_iter()
            .map(|model| Self::normalize_model_id(&model))
            .filter(|model| !model.is_empty())
            .collect();
        (!models.is_empty()).then_some(models)
    }

    pub(crate) fn resolve_available_models(
        native_output: Option<&str>,
        materialized_models: Option<&str>,
    ) -> Vec<String> {
        let native_models: Vec<String> = native_output
            .map(|output| {
                output
                    .lines()
                    .map(str::trim)
                    .filter(|line| {
                        !line.is_empty()
                            && !line.starts_with("Usage:")
                            && !line.starts_with("List available")
                            && !line.starts_with("Flags:")
                            && !line.starts_with("-h")
                            && !line.starts_with("--help")
                    })
                    .map(Self::normalize_model_id)
                    .collect()
            })
            .unwrap_or_default();
        let cached_models = materialized_models
            .and_then(Self::parse_materialized_models)
            .unwrap_or_default();

        let mut seen = HashSet::new();
        (if native_models.is_empty() {
            cached_models
        } else {
            native_models
        })
        .into_iter()
        .filter(|model| seen.insert(model.clone()))
        .collect()
    }

    /// Discover through the controlled Agy boundary. The materialized JSON is
    /// only an ordered fallback/cache and can never suppress a newer native
    /// entry. Native order is significant: its first entry is Agy's default.
    fn fetch_available_models(allow_materialized_fallback: bool) -> Vec<String> {
        let cached = allow_materialized_fallback
            .then(|| std::env::var("AGY_ACP_MODELS_JSON").ok())
            .flatten();
        if cached
            .as_deref()
            .is_some_and(|raw| Self::parse_materialized_models(raw).is_none())
        {
            eprintln!("[agy-acp] ignoring invalid AGY_ACP_MODELS_JSON fallback");
        }

        let discovery_bin =
            std::env::var("AGY_MODEL_DISCOVERY_BIN").unwrap_or_else(|_| "agy".to_string());
        let timeout_ms = std::env::var("AGY_MODEL_DISCOVERY_TIMEOUT_MS")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(3500);
        let native_output =
            Self::run_model_discovery_bounded(&discovery_bin, Duration::from_millis(timeout_ms));
        if native_output.is_none() {
            eprintln!("[agy-acp] native model discovery unavailable; using materialized fallback");
        }

        let models = Self::resolve_available_models(native_output.as_deref(), cached.as_deref());
        if models.is_empty() {
            eprintln!("[agy-acp] no model catalog available; deferring to Agy native default");
        }
        models
    }

    fn refresh_models_for_account(&mut self, new_session: bool) {
        let account = crate::account::active_credential_fingerprint();
        if account == self.model_cache.account
            && !self.available_models.is_empty()
            && (!new_session || account.is_some())
        {
            return;
        }
        let discovery_bin =
            std::env::var("AGY_MODEL_DISCOVERY_BIN").unwrap_or_else(|_| "agy".to_string());
        let timeout_ms = std::env::var("AGY_MODEL_DISCOVERY_TIMEOUT_MS")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(3500);
        let native_output =
            Self::run_model_discovery_bounded(&discovery_bin, Duration::from_millis(timeout_ms));
        let discovered = Self::resolve_available_models(native_output.as_deref(), None);
        self.available_models = self.model_cache.update(account, discovered);
    }

    fn run_model_discovery_bounded(discovery_bin: &str, timeout: Duration) -> Option<String> {
        use std::io::Read;
        use std::process::Stdio;
        let mut child = std::process::Command::new(discovery_bin)
            .arg("models")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdout_pipe = child.stdout.take()?;
        let reader_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut pipe = stdout_pipe;
            let _ = pipe.read_to_end(&mut buf);
            String::from_utf8_lossy(&buf).into_owned()
        });
        let start = std::time::Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        return None;
                    }
                    return reader_handle.join().ok();
                }
                Ok(None) => {
                    if start.elapsed() >= timeout {
                        #[cfg(windows)]
                        {
                            let _ = std::process::Command::new("taskkill")
                                .args(["/F", "/T", "/PID", &child.id().to_string()])
                                .stdin(Stdio::null())
                                .stdout(Stdio::null())
                                .stderr(Stdio::null())
                                .status();
                        }
                        let _ = child.kill();
                        let _ = child.wait();
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(40));
                }
                Err(_) => return None,
            }
        }
    }

    fn models_for_session(&mut self, model_id: Option<&str>) -> Vec<String> {
        self.refresh_models_for_account(false);
        let mut models = self.available_models.clone();
        if let Some(current) = model_id
            .filter(|model| !model.trim().is_empty())
            .map(|model| self.resolve_session_model_id(model))
        {
            if !models.iter().any(|model| model == &current) {
                models.push(current);
            }
        }
        models
    }

    /// Build the ACP `models` JSON for a session, given its current model_id.
    pub fn session_models_json(&mut self, model_id: Option<&str>) -> Value {
        let resolved_model = model_id
            .filter(|model| !model.trim().is_empty())
            .map(|model| self.resolve_session_model_id(model));
        let models = self.models_for_session(resolved_model.as_deref());
        let current = resolved_model
            .as_deref()
            .or_else(|| models.first().map(|s| s.as_str()))
            .unwrap_or("");
        let available: Vec<Value> = models
            .iter()
            .map(|name| {
                json!({
                    "modelId": name,
                    "name": name,
                })
            })
            .collect();
        json!({
            "currentModelId": current,
            "availableModels": available,
        })
    }

    /// Build the ACP session config option that Zed uses for its model selector.
    #[cfg(test)]
    pub fn session_config_options_json(&mut self, model_id: Option<&str>) -> Value {
        self.session_config_options_json_for_mode(model_id, DEFAULT_MODE_ID)
    }

    pub fn session_config_options_json_for_mode(
        &mut self,
        model_id: Option<&str>,
        mode_id: &str,
    ) -> Value {
        let resolved_model = model_id
            .filter(|model| !model.trim().is_empty())
            .map(|model| self.resolve_session_model_id(model));
        let models = self.models_for_session(resolved_model.as_deref());
        let current = resolved_model
            .as_deref()
            .or_else(|| models.first().map(|s| s.as_str()))
            .unwrap_or("");
        let options: Vec<Value> = models
            .iter()
            .map(|name| {
                json!({
                    "value": name,
                    "name": name,
                })
            })
            .collect();
        let mut mode_options = vec![
            json!({ "value": Self::ACCEPT_EDITS_MODE_ID, "name": "Accept edits" }),
            json!({ "value": Self::PLAN_MODE_ID, "name": "Plan" }),
        ];
        if self.allow_full_access {
            mode_options.push(json!({ "value": Self::FULL_ACCESS_MODE_ID, "name": "Full access" }));
        }
        json!([
            {
                "id": Self::MODEL_CONFIG_ID,
                "name": "Model",
                "category": "model",
                "type": "select",
                "currentValue": current,
                "options": options,
            },
            {
                "id": Self::MODE_CONFIG_ID,
                "name": "Mode",
                "category": "mode",
                "type": "select",
                "currentValue": mode_id,
                "options": mode_options,
            }
        ])
    }

    pub fn session_config_result_json(
        &mut self,
        session_id: &str,
        model_id: Option<&str>,
    ) -> Value {
        self.refresh_models_for_account(true);
        let mode_id = self
            .sessions
            .get(session_id)
            .map(|session| session.mode_id.as_str())
            .unwrap_or(DEFAULT_MODE_ID)
            .to_string();
        json!({
            "sessionId": session_id,
            "models": self.session_models_json(model_id),
            "configOptions": self.session_config_options_json_for_mode(model_id, &mode_id),
        })
    }

    /// Acquire exclusive lock on a dedicated lock file for read-write mutual exclusion.
    fn lock_state_file(&self) -> Option<fs::File> {
        if let Some(parent) = self.state_file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let lock_path = self.state_file.with_extension("lock");
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .ok()?;
        lock_file.lock_exclusive().ok()?;
        Some(lock_file)
    }

    /// Load persisted session store (caller must hold lock).
    fn load_store_inner(&self) -> SessionStore {
        let Some(file) = fs::File::open(&self.state_file).ok() else {
            return SessionStore::default();
        };
        serde_json::from_reader(&file).unwrap_or_default()
    }

    /// Load persisted session store with lock.
    pub fn load_store(&self) -> SessionStore {
        let _lock = self.lock_state_file();
        self.load_store_inner()
    }

    fn sanitize_context_for_persistence(value: &Value) -> Value {
        match value {
            Value::Object(fields) => {
                let mut sanitized = Map::new();
                for (key, value) in fields {
                    if key.eq_ignore_ascii_case("mcpServers")
                        || key.eq_ignore_ascii_case("mcp_servers")
                    {
                        if !Self::mcp_servers_are_nonempty(value) {
                            sanitized.insert(key.clone(), json!([]));
                        }
                        continue;
                    }
                    sanitized.insert(key.clone(), Self::sanitize_context_for_persistence(value));
                }
                Value::Object(sanitized)
            }
            Value::Array(values) => Value::Array(
                values
                    .iter()
                    .map(Self::sanitize_context_for_persistence)
                    .collect(),
            ),
            _ => value.clone(),
        }
    }

    /// Try to restore conversation_id, last_step_idx, model_id, caller
    /// context, and permission mode from persisted state.
    #[cfg(test)]
    pub fn restore_session(&self, session_id: &str) -> Option<(String, i64, Option<String>)> {
        let store = self.load_store();
        store.sessions.get(session_id).and_then(|s| {
            s.conversation_id
                .clone()
                .map(|cid| (cid, s.last_step_idx, s.model_id.clone()))
        })
    }

    /// Persist a session binding (read-modify-write under single lock).
    pub fn persist_session(
        &self,
        session_id: &str,
        conversation_id: Option<&str>,
        last_step_idx: i64,
        model_id: Option<&str>,
    ) {
        let Some(_lock) = self.lock_state_file() else {
            return;
        };
        let session = self.sessions.get(session_id);
        if session
            .map(|s| Self::session_has_required_mcp(s, self.allow_mcp))
            .unwrap_or(false)
        {
            return;
        }
        let mode_id = session
            .map(|session| session.mode_id.clone())
            .unwrap_or_else(default_mode_id);
        let mode_id = if mode_id == Self::FULL_ACCESS_MODE_ID && !self.allow_full_access {
            Self::PLAN_MODE_ID.to_string()
        } else if mode_id == Self::ACCEPT_EDITS_MODE_ID
            && !self.worker_write_authorized_for_session(session_id)
        {
            Self::PLAN_MODE_ID.to_string()
        } else {
            mode_id
        };
        if !self.is_supported_mode(&mode_id) {
            return;
        }
        let caller_context = session
            .map(|session| session.caller_context.clone())
            .unwrap_or_else(|| json!({}));
        let caller_context = Self::sanitize_context_for_persistence(&caller_context);
        let mut store = self.load_store_inner();
        store.sessions.insert(
            session_id.to_string(),
            StoredSession {
                conversation_id: conversation_id.map(String::from),
                last_step_idx,
                model_id: model_id.map(String::from),
                mode_id,
                caller_context,
                mcp_servers: session
                    .filter(|session| !Self::mcp_servers_are_nonempty(&session.mcp_servers))
                    .map(|session| session.mcp_servers.clone())
                    .unwrap_or_else(|| json!([])),
            },
        );
        let tmp = self.state_file.with_extension("tmp");
        if let Ok(file) = fs::File::create(&tmp) {
            if serde_json::to_writer_pretty(&file, &store).is_ok() {
                let _ = fs::rename(&tmp, &self.state_file);
            }
        }
    }

    pub fn read_replay_updates_from_db_inner(
        &self,
        conversation_id: &str,
    ) -> Option<(Vec<Value>, i64)> {
        read_replay_updates_from_db(&self.conversations_dir, conversation_id, self.skip_naration)
    }

    #[cfg(test)]
    fn read_delta_from_db_inner(
        &self,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<crate::types::ConversationDelta> {
        read_delta_from_db(&self.conversations_dir, conversation_id, after_step_idx)
    }

    #[cfg(test)]
    pub fn read_response_from_db(
        &self,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<(String, i64)> {
        self.read_delta_from_db_inner(conversation_id, after_step_idx)
            .and_then(|delta| delta.text.map(|text| (text, delta.max_step_idx)))
    }

    /// Filter out leading narration ("I will ...", "I'll ...") from response parts.
    #[cfg(test)]
    pub fn filter_narration(parts: &[String]) -> Option<String> {
        filter_narration(parts)
    }

    /// A part is considered narration if every non-empty line starts with "I will" or "I'll".
    #[cfg(test)]
    pub fn is_narration(text: &str) -> bool {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return false;
        }
        lines.iter().all(|l| {
            let line = l.trim_start();
            line.starts_with("I will") || line.starts_with("I'll") || line.starts_with("I’ll")
        })
    }

    fn evict_if_needed(&mut self) {
        const MAX_SESSIONS: usize = 64;
        while self.sessions.len() >= MAX_SESSIONS {
            if let Some(key) = self.sessions.keys().next().cloned() {
                self.sessions.remove(&key);
                self.mcp_registry.remove(&key);
            }
        }
    }

    fn restore_session_state_checked(
        &mut self,
        session_id: &str,
    ) -> Result<bool, SessionRequestError> {
        let store = self.load_store();
        let Some(stored) = store.sessions.get(session_id) else {
            return Ok(false);
        };
        if Self::stored_session_has_required_mcp(stored, self.allow_mcp) {
            return Err(SessionRequestError::McpUnavailable);
        }
        let mut mode_id = stored.mode_id.clone();
        if mode_id == Self::FULL_ACCESS_MODE_ID && !self.allow_full_access {
            // Legacy delegated sessions may have recorded full-access before
            // the worker capability boundary existed. Restoring them as plan
            // preserves the conversation without granting a write capability.
            mode_id = Self::PLAN_MODE_ID.to_string();
        } else if mode_id == Self::ACCEPT_EDITS_MODE_ID
            && !Self::stored_session_assignment_is_approved(stored)
        {
            mode_id = Self::PLAN_MODE_ID.to_string();
        }
        if !self.is_supported_mode(&mode_id) {
            return Err(SessionRequestError::Invalid(
                "stored session has an unsupported permission mode".to_string(),
            ));
        }
        if !self.sessions.contains_key(session_id) {
            self.evict_if_needed();
        }
        self.sessions.insert(
            session_id.to_string(),
            Session {
                conversation_id: stored.conversation_id.clone(),
                last_step_idx: stored.last_step_idx,
                model_id: stored.model_id.clone(),
                mode_id,
                caller_context: stored.caller_context.clone(),
                mcp_servers: stored.mcp_servers.clone(),
            },
        );
        Ok(true)
    }

    pub fn restore_session_state(&mut self, session_id: &str) -> bool {
        matches!(self.restore_session_state_checked(session_id), Ok(true))
    }

    pub fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        let available: Vec<Value> = self
            .available_models
            .iter()
            .map(|name| {
                json!({
                    "modelId": name,
                    "name": name,
                })
            })
            .collect();

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": 1,
                "agentInfo": { "name": "agy", "version": env!("CARGO_PKG_VERSION") },
                "agentCapabilities": {
                    "loadSession": true,
                    "promptCapabilities": { "image": true },
                    "sessionCapabilities": { "resume": {} },
                },
                "models": {
                    "currentModelId": self.available_models.first().map(|s| s.as_str()).unwrap_or("Gemini 4 Argon (High)"),
                    "availableModels": available,
                },
                "authMethods": [],
            })),
            error: None,
        }
    }

    pub fn handle_session_new(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let (mcp_servers, mode_id) = match self.parse_session_params(params, false) {
            Ok(request) => request,
            Err(error) => return Self::session_request_error_response(id, error),
        };
        let assignment = match Self::assignment_from_request(params) {
            Ok(assignment) => assignment,
            Err(error) => return Self::error_response(id, AGY_WORKER_ERROR_CODE, error),
        };
        let session_id = Uuid::new_v4().to_string();
        self.evict_if_needed();
        if let Err(error) = self.attach_mcp_for_session(&session_id, &mcp_servers) {
            return Self::session_request_error_response(id, error);
        }
        // Preserve ordinary caller context, but keep assignment aliases out
        // of it until `remember_worker_assignment` records the canonical
        // `_agyWorker` state. This lets the adapter distinguish a new Lead
        // approval from an already-approved session.
        let mut caller_context = json!({});
        Self::merge_context(&mut caller_context, params);
        self.sessions.insert(
            session_id.clone(),
            Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: None,
                mode_id: mode_id.unwrap_or_else(default_mode_id),
                caller_context,
                mcp_servers,
            },
        );
        if let Some(assignment) = assignment.as_ref() {
            if let Err(error) = self.remember_worker_assignment(&session_id, assignment) {
                self.sessions.remove(&session_id);
                self.mcp_registry.remove(&session_id);
                return Self::error_response(id, AGY_WORKER_ERROR_CODE, error);
            }
        }
        self.normalize_worker_mode(&session_id);
        self.persist_session(&session_id, None, -1, None);
        let result = self.session_config_result_json(&session_id, None);
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn handle_session_load(&mut self, id: Value, params: &Value) -> Vec<String> {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return vec![serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId"})),
            })
            .unwrap()];
        }

        let (mcp_servers, mode_id) = match self.parse_session_params(params, true) {
            Ok(request) => request,
            Err(error) => {
                return vec![serde_json::to_string(&Self::session_request_error_response(
                    id, error,
                ))
                .unwrap()]
            }
        };
        let assignment = match Self::assignment_from_request(params) {
            Ok(assignment) => assignment,
            Err(error) => {
                return vec![serde_json::to_string(&Self::error_response(
                    id,
                    AGY_WORKER_ERROR_CODE,
                    error,
                ))
                .unwrap()]
            }
        };

        let was_in_memory = self.sessions.contains_key(session_id);
        if !was_in_memory {
            match self.restore_session_state_checked(session_id) {
                Ok(true) => {}
                Ok(false) => {
                    return vec![serde_json::to_string(&Self::error_response(
                        id,
                        -32000,
                        format!("unknown sessionId: {session_id}"),
                    ))
                    .unwrap()]
                }
                Err(error) => {
                    return vec![serde_json::to_string(&Self::session_request_error_response(
                        id, error,
                    ))
                    .unwrap()]
                }
            }
        }

        if let Err(error) = self.attach_mcp_for_session(session_id, &mcp_servers) {
            if !was_in_memory {
                self.sessions.remove(session_id);
                self.mcp_registry.remove(session_id);
            }
            return vec![
                serde_json::to_string(&Self::session_request_error_response(id, error)).unwrap(),
            ];
        }

        if let Err(error) =
            self.update_session_from_request(session_id, params, &mcp_servers, mode_id.as_deref())
        {
            if !was_in_memory {
                self.sessions.remove(session_id);
                self.mcp_registry.remove(session_id);
            }
            return vec![
                serde_json::to_string(&Self::session_request_error_response(id, error)).unwrap(),
            ];
        }
        if let Some(assignment) = assignment.as_ref() {
            if let Err(error) = self.remember_worker_assignment(session_id, assignment) {
                if !was_in_memory {
                    self.sessions.remove(session_id);
                    self.mcp_registry.remove(session_id);
                }
                return vec![serde_json::to_string(&Self::error_response(
                    id,
                    AGY_WORKER_ERROR_CODE,
                    error,
                ))
                .unwrap()];
            }
        }
        self.normalize_worker_mode(session_id);

        let mut output_lines: Vec<String> = Vec::new();

        let replay_conv_id = self
            .sessions
            .get(session_id)
            .and_then(|session| session.conversation_id.clone());
        if let Some(conv_id) = replay_conv_id {
            if let Some((updates, max_step_idx)) = self.read_replay_updates_from_db_inner(&conv_id)
            {
                for update in updates {
                    let notification = serde_json::to_string(&JsonRpcNotification {
                        jsonrpc: "2.0",
                        method: "session/update".to_string(),
                        params: json!({
                            "sessionId": session_id,
                            "update": update,
                        }),
                    })
                    .unwrap();
                    output_lines.push(notification);
                }
                if let Some(session) = self.sessions.get_mut(session_id) {
                    session.last_step_idx = max_step_idx;
                }
                let model_id = self
                    .sessions
                    .get(session_id)
                    .and_then(|s| s.model_id.clone());
                self.persist_session(
                    session_id,
                    Some(conv_id.as_str()),
                    max_step_idx,
                    model_id.as_deref(),
                );
            }
        }

        if let Some(session) = self.sessions.get(session_id) {
            let model_id = session.model_id.clone();
            let conversation_id = session.conversation_id.clone();
            let last_step_idx = session.last_step_idx;
            self.persist_session(
                session_id,
                conversation_id.as_deref(),
                last_step_idx,
                model_id.as_deref(),
            );
        }

        output_lines.push({
            let model_id = self
                .sessions
                .get(session_id)
                .and_then(|s| s.model_id.clone());
            let result = self.session_config_result_json(session_id, model_id.as_deref());
            serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(result),
                error: None,
            })
            .unwrap()
        });

        output_lines
    }

    pub fn handle_session_resume(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId"})),
            };
        }

        let (mcp_servers, mode_id) = match self.parse_session_params(params, true) {
            Ok(request) => request,
            Err(error) => return Self::session_request_error_response(id, error),
        };
        let assignment = match Self::assignment_from_request(params) {
            Ok(assignment) => assignment,
            Err(error) => return Self::error_response(id, AGY_WORKER_ERROR_CODE, error),
        };

        let was_in_memory = self.sessions.contains_key(session_id);
        if !was_in_memory {
            match self.restore_session_state_checked(session_id) {
                Ok(true) => {}
                Ok(false) => {
                    return Self::error_response(
                        id,
                        -32000,
                        format!("unknown sessionId: {session_id}"),
                    )
                }
                Err(error) => return Self::session_request_error_response(id, error),
            }
        }

        if let Err(error) = self.attach_mcp_for_session(session_id, &mcp_servers) {
            if !was_in_memory {
                self.sessions.remove(session_id);
                self.mcp_registry.remove(session_id);
            }
            return Self::session_request_error_response(id, error);
        }

        if let Err(error) =
            self.update_session_from_request(session_id, params, &mcp_servers, mode_id.as_deref())
        {
            if !was_in_memory {
                self.sessions.remove(session_id);
                self.mcp_registry.remove(session_id);
            }
            return Self::session_request_error_response(id, error);
        }
        if let Some(assignment) = assignment.as_ref() {
            if let Err(error) = self.remember_worker_assignment(session_id, assignment) {
                if !was_in_memory {
                    self.sessions.remove(session_id);
                    self.mcp_registry.remove(session_id);
                }
                return Self::error_response(id, AGY_WORKER_ERROR_CODE, error);
            }
        }
        self.normalize_worker_mode(session_id);

        let (model_id, conversation_id, last_step_idx) = {
            let session = self.sessions.get(session_id).expect("session was restored");
            (
                session.model_id.clone(),
                session.conversation_id.clone(),
                session.last_step_idx,
            )
        };
        self.persist_session(
            session_id,
            conversation_id.as_deref(),
            last_step_idx,
            model_id.as_deref(),
        );
        let result = self.session_config_result_json(session_id, model_id.as_deref());
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn handle_session_set_model(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        self.refresh_models_for_account(false);
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let model_id = params.get("modelId").and_then(|v| v.as_str()).unwrap_or("");

        if session_id.is_empty() || model_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId or modelId"})),
            };
        }

        if !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id);
        }

        let normalized_model = self.resolve_session_model_id(model_id);
        if let Some(assignment) = self
            .sessions
            .get(session_id)
            .and_then(Self::session_assignment)
        {
            if Self::assignment_is_approved(&assignment)
                && assignment
                    .get("modelId")
                    .and_then(Value::as_str)
                    .map(|m| self.resolve_session_model_id(m))
                    .as_deref()
                    .is_some_and(|assigned| assigned != normalized_model.as_str())
            {
                return Self::error_response(
                    id,
                    AGY_WORKER_ERROR_CODE,
                    "approved Agy assignment is immutable; model changes require a new Lead task",
                );
            }
        }

        let Some(session) = self.sessions.get_mut(session_id) else {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32000,
                    "message": format!("unknown sessionId: {session_id}"),
                })),
            };
        };

        session.model_id = Some(normalized_model);
        let model_id_str = session.model_id.clone();
        let last_step_idx = session.last_step_idx;
        let conv_id = session.conversation_id.clone();

        self.persist_session(
            session_id,
            conv_id.as_deref(),
            last_step_idx,
            model_id_str.as_deref(),
        );

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({})),
            error: None,
        }
    }

    pub fn handle_session_set_config_option(
        &mut self,
        id: Value,
        params: &Value,
    ) -> JsonRpcResponse {
        self.refresh_models_for_account(false);
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let config_id = params
            .get("configId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let value = params.get("value").and_then(|v| v.as_str()).unwrap_or("");

        if session_id.is_empty() || config_id.is_empty() || value.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(
                    json!({"code":-32602,"message":"missing sessionId, configId, or value"}),
                ),
            };
        }
        if config_id != Self::MODEL_CONFIG_ID && config_id != Self::MODE_CONFIG_ID {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32602,
                    "message": format!("unknown configId: {config_id}"),
                })),
            };
        }
        if config_id == Self::MODE_CONFIG_ID && !self.is_supported_mode(value) {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32602,
                    "message": format!("unknown mode: {value}"),
                })),
            };
        }

        if !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id);
        }

        let resolved_config_model =
            (config_id == Self::MODEL_CONFIG_ID).then(|| self.resolve_session_model_id(value));
        if let Some(normalized_model) = resolved_config_model.as_ref() {
            if let Some(assignment) = self
                .sessions
                .get(session_id)
                .and_then(Self::session_assignment)
            {
                if Self::assignment_is_approved(&assignment)
                    && assignment
                        .get("modelId")
                        .and_then(Value::as_str)
                        .map(|m| self.resolve_session_model_id(m))
                        .as_deref()
                        .is_some_and(|assigned| assigned != normalized_model.as_str())
                {
                    return Self::error_response(
                        id,
                        AGY_WORKER_ERROR_CODE,
                        "approved Agy assignment is immutable; model changes require a new Lead task",
                    );
                }
            }
        }

        let direct_unassigned_mode = config_id == Self::MODE_CONFIG_ID
            && value == Self::ACCEPT_EDITS_MODE_ID
            && !self.worker_assignment_approved(session_id)
            && self
                .sessions
                .get(session_id)
                .map(|session| {
                    // A direct Paseo session has no Lead envelope. Its host
                    // mode selection is the user-controlled, permission-
                    // bounded path. Any assignment marker (valid or malformed)
                    // remains fail-closed until Lead approval is present.
                    !Self::assignment_key_present(&session.caller_context)
                })
                .unwrap_or(false);
        if config_id == Self::MODE_CONFIG_ID
            && value == Self::ACCEPT_EDITS_MODE_ID
            && !self.worker_assignment_approved(session_id)
            && !direct_unassigned_mode
        {
            return Self::error_response(id, AGY_WORKER_ERROR_CODE, AGY_WORKER_BLOCKED);
        }

        let Some(session) = self.sessions.get_mut(session_id) else {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32000,
                    "message": format!("unknown sessionId: {session_id}"),
                })),
            };
        };

        if let Some(normalized_model) = resolved_config_model {
            session.model_id = Some(normalized_model);
        } else {
            session.mode_id = value.to_string();
        }
        let model_id = session.model_id.clone();
        let mode_id = session.mode_id.clone();
        let last_step_idx = session.last_step_idx;
        let conv_id = session.conversation_id.clone();

        self.persist_session(
            session_id,
            conv_id.as_deref(),
            last_step_idx,
            model_id.as_deref(),
        );

        let config_options =
            self.session_config_options_json_for_mode(model_id.as_deref(), &mode_id);
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "configOptions": config_options })),
            error: None,
        }
    }

    pub fn handle_session_set_mode(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        let mode_id = params
            .get("modeId")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        let config_params = json!({
            "sessionId": session_id,
            "configId": Self::MODE_CONFIG_ID,
            "value": mode_id,
        });
        let mut response = self.handle_session_set_config_option(id, &config_params);
        if response.error.is_none() {
            response.result = Some(json!({}));
        }
        response
    }

    fn session_working_dir(&self, session_id: &str) -> String {
        self.sessions
            .get(session_id)
            .and_then(|session| session.caller_context.get("cwd"))
            .and_then(Value::as_str)
            .filter(|cwd| !cwd.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.working_dir.clone())
    }

    pub fn build_agy_args(
        &self,
        session_id: &str,
        run_log_path: &Path,
        attachment_dir: Option<&Path>,
    ) -> Vec<String> {
        let effective_mode = self.effective_agy_mode(session_id);
        self.build_agy_args_for_mode(
            session_id,
            run_log_path,
            attachment_dir,
            Some(effective_mode.as_str()),
        )
    }

    fn build_agy_args_for_mode(
        &self,
        session_id: &str,
        run_log_path: &Path,
        attachment_dir: Option<&Path>,
        mode_override: Option<&str>,
    ) -> Vec<String> {
        let mut args = vec![
            "--log-file".to_string(),
            run_log_path.to_string_lossy().to_string(),
            "--print-timeout".to_string(),
            std::env::var("AGY_PRINT_TIMEOUT")
                .unwrap_or_else(|_| DEFAULT_PRINT_TIMEOUT.to_string()),
            "--input-format".to_string(),
            "stream-json".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
        ];
        if let Some(path) = attachment_dir {
            args.push("--add-dir".to_string());
            args.push(path.to_string_lossy().to_string());
        }
        if let Ok(extra) = std::env::var("AGY_EXTRA_ARGS") {
            let mut skip_value = false;
            for extra_arg in extra.split_whitespace() {
                if skip_value {
                    skip_value = false;
                    continue;
                }
                // Provider configuration must not override the adapter's
                // worker identity, selected model/conversation, permission
                // mode, or safety boundary.  Those values come from the
                // session and the trusted launcher only.
                if matches!(
                    extra_arg,
                    "--mode"
                        | "--model"
                        | "--conversation"
                        | "--agent"
                        | "--dangerously-skip-permissions"
                        | "--yolo"
                        | "--auto-approve"
                        | "--no-confirm"
                ) {
                    skip_value = matches!(
                        extra_arg,
                        "--mode" | "--model" | "--conversation" | "--agent"
                    );
                    continue;
                }
                if extra_arg.starts_with("--mode=")
                    || extra_arg.starts_with("--model=")
                    || extra_arg.starts_with("--conversation=")
                    || extra_arg.starts_with("--agent=")
                    || extra_arg == "--dangerously-skip-permissions"
                    || extra_arg == "--yolo"
                    || extra_arg == "--auto-approve"
                    || extra_arg == "--no-confirm"
                {
                    continue;
                }
                args.push(extra_arg.to_string());
            }
        }
        let worker_agent = std::env::var("AGY_AGENT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| AGY_WORKER_AGENT.to_string());
        args.push("--agent".to_string());
        args.push(worker_agent);
        let requested_mode_id = self
            .sessions
            .get(session_id)
            .map(|session| session.mode_id.as_str())
            .unwrap_or(DEFAULT_MODE_ID);
        let mode_id = mode_override.unwrap_or(requested_mode_id);
        // Agy 1.1.21 accepts only native `accept-edits` and `plan` modes.
        // The ACP-only `full-access` mode is available only when the trusted
        // process launcher granted the autonomous-root capability. Session
        // parameters and persisted state cannot manufacture that capability.
        let agy_mode_id = if mode_id == FULL_ACCESS_MODE_ID && self.allow_full_access {
            Self::ACCEPT_EDITS_MODE_ID
        } else if self.is_supported_mode(mode_id) {
            mode_id
        } else {
            // A malformed in-memory state must never gain permissions.
            Self::PLAN_MODE_ID
        };
        args.push("--mode".to_string());
        args.push(agy_mode_id.to_string());
        // Agy headless print mode currently auto-denies every tool that would
        // normally show a confirmation, including read-only ListDir. The
        // native skip flag is therefore required for an explicitly authorized
        // accept-edits turn to function at all. Plan never receives it, and an
        // unapproved delegated assignment cannot reach effective accept-edits.
        let skip_permissions = agy_mode_id == Self::ACCEPT_EDITS_MODE_ID
            && (self.worker_write_authorized_for_session(session_id)
                || (requested_mode_id == FULL_ACCESS_MODE_ID && self.allow_full_access)
                || (mode_id == FULL_ACCESS_MODE_ID && self.allow_full_access));
        if skip_permissions {
            args.push("--dangerously-skip-permissions".to_string());
        }
        if let Some(session) = self.sessions.get(session_id) {
            if let Some(conv_id) = &session.conversation_id {
                args.push("--conversation".to_string());
                args.push(conv_id.clone());
            }
            if let Some(model_id) = &session.model_id {
                args.push("--model".to_string());
                args.push(self.resolve_session_model_id(model_id));
            }
        }
        args
    }

    pub fn build_stream_input_line(prompt: &str) -> Result<String, serde_json::Error> {
        let prompt = Self::with_worker_contract(prompt, None, Self::PLAN_MODE_ID, true);
        serde_json::to_string(&json!({
            "event": "user",
            "message": { "content": prompt },
        }))
    }

    fn is_supervisor_runtime() -> bool {
        std::env::var("AGY_RUNTIME_ROLE")
            .map(|value| value.eq_ignore_ascii_case("supervisor"))
            .unwrap_or(false)
            || std::env::var("AGY_AGENT")
                .map(|value| value.eq_ignore_ascii_case("agy-supervisor"))
                .unwrap_or(false)
    }

    fn supervisor_checkpoint_path(session_id: &str) -> Result<PathBuf, String> {
        // Checkpoint state is owned by one ACP session. A provider-wide file
        // lets concurrent Supervisor seats overwrite and ingest each other's
        // human context.
        Uuid::parse_str(session_id).map_err(|_| "invalid ACP session identity".to_string())?;
        let root = std::env::var("AGY_SUPERVISOR_CHECKPOINT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME")
                    .or_else(|_| std::env::var("USERPROFILE"))
                    .unwrap_or_else(|_| ".".to_string());
                PathBuf::from(home)
                    .join(".config/room-workflow/supervisor-checkpoints/agy-supervisor")
            });
        Ok(root.join(format!("{session_id}.json")))
    }

    fn normalize_workspace_path(path: &str) -> String {
        path.trim()
            .trim_end_matches(['\\', '/'])
            .replace('/', "\\")
            .to_lowercase()
    }

    pub(crate) fn checkpoint_matches_workspace(value: &Value, expected_workspace: &str) -> bool {
        value
            .pointer("/project/workspacePath")
            .and_then(Value::as_str)
            .map(|checkpoint_workspace| {
                Self::normalize_workspace_path(expected_workspace)
                    == Self::normalize_workspace_path(checkpoint_workspace)
            })
            .unwrap_or(true)
    }

    fn supervisor_checkpoint_context(
        session_id: Option<&str>,
        expected_workspace: Option<&str>,
    ) -> String {
        let Some(session_id) = session_id else {
            return "<supervisor-checkpoint state=\"unbound\" />".to_string();
        };
        let path = match Self::supervisor_checkpoint_path(session_id) {
            Ok(path) => path,
            Err(_) => return "<supervisor-checkpoint state=\"invalid-session\" />".to_string(),
        };
        let display_path = path.display();
        let Ok(raw) = fs::read_to_string(&path) else {
            return format!(
                "<supervisor-checkpoint state=\"missing\" sessionId=\"{session_id}\" path=\"{display_path}\" />"
            );
        };
        if ["Ã", "Ä", "Æ", "áº", "á»"]
            .iter()
            .any(|marker| raw.contains(marker))
        {
            return format!(
                "<supervisor-checkpoint state=\"invalid-encoding\" sessionId=\"{session_id}\" path=\"{display_path}\" />"
            );
        }
        let Ok(value) = serde_json::from_str::<Value>(&raw) else {
            return format!(
                "<supervisor-checkpoint state=\"invalid\" sessionId=\"{session_id}\" path=\"{display_path}\" />"
            );
        };
        if let Some(expected_workspace) = expected_workspace {
            if !Self::checkpoint_matches_workspace(&value, expected_workspace) {
                return format!(
                    "<supervisor-checkpoint state=\"scope-mismatch\" sessionId=\"{session_id}\" expectedWorkspace=\"{expected_workspace}\" />"
                );
            }
        }
        let normalized = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
        format!(
            "<supervisor-checkpoint state=\"loaded\" sessionId=\"{session_id}\" path=\"{display_path}\">\n{normalized}\n</supervisor-checkpoint>"
        )
    }

    pub(crate) fn has_slp_governance_channel(mcp_servers: &Value) -> bool {
        if let Some(servers) = mcp_servers.as_array() {
            return servers.iter().any(|server| {
                server
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| name == "slp-governance")
                    .unwrap_or(false)
            });
        }
        if let Some(map) = mcp_servers.as_object() {
            return map.contains_key("slp-governance");
        }
        false
    }

    pub(crate) fn parse_seat_defaults_env(raw: &str) -> Option<(String, String)> {
        let value: Value = serde_json::from_str(raw).ok()?;
        let lead = value.get("lead")?.as_str()?.trim().to_string();
        let peer = value.get("peer")?.as_str()?.trim().to_string();
        if lead.is_empty() || peer.is_empty() {
            return None;
        }
        Some((lead, peer))
    }

    pub(crate) fn supervisor_contract_prompt(
        prompt: &str,
        session_id: Option<&str>,
        expected_workspace: Option<&str>,
        slp_channel: bool,
    ) -> String {
        // With the SLP governance channel staged, the durable checkpoint lives
        // in the MCP checkpoint store; reading the on-disk JSON here would
        // inject a stale copy by design.
        let checkpoint = if slp_channel {
            String::new()
        } else {
            Self::supervisor_checkpoint_context(session_id, expected_workspace)
        };
        // Owner-configured Lead/Peer defaults arrive per session from the SLP
        // plugin via process env (one adapter process per seat); absent or
        // malformed values keep the legacy block byte-identical.
        let seat_defaults = std::env::var("SLP_SEAT_DEFAULTS_JSON")
            .ok()
            .and_then(|raw| Self::parse_seat_defaults_env(&raw));
        Self::supervisor_runtime_block(
            prompt,
            &checkpoint,
            slp_channel,
            seat_defaults
                .as_ref()
                .map(|(lead, peer)| (lead.as_str(), peer.as_str())),
        )
    }

    pub(crate) fn supervisor_runtime_block(
        prompt: &str,
        checkpoint: &str,
        slp_channel: bool,
        seat_defaults: Option<(&str, &str)>,
    ) -> String {
        let checkpoint_update_rule = if slp_channel {
            "After a durable owner decision or correction, update the complete checkpoint atomically through the `slp_checkpoint_write` tool of the `slp-governance` MCP server (read the current value first with `slp_checkpoint_read`); write the full JSON object, including `currentSubMission` and an `updatedAt` UTC timestamp. If those checkpoint tools are unavailable or a write fails, keep responsibility state in the durable worker contract and do not fall back to workspace files or scripts.".to_string()
        } else {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".to_string());
            let script =
                PathBuf::from(home).join(".config/room-workflow/Update-SupervisorCheckpoint.ps1");
            format!("Update the complete checkpoint atomically through `{}` after a durable owner decision or correction; the adapter supplies the active session identity.", script.display())
        };
        let checkpoint_section = if slp_channel {
            "The durable Supervisor checkpoint lives in the `slp-governance` MCP checkpoint store; read it with `slp_checkpoint_read` at turn start and after any uncertainty instead of relying on injected copies."
        } else {
            checkpoint
        };
        let seat_defaults_line = match seat_defaults {
            Some((lead, peer)) => format!(
                " Owner-configured seat defaults: Lead `{lead}`, Peer `{peer}`. Provider selection order when dispatching or consulting: (1) a provider the owner explicitly names in the current task, (2) these defaults, which supersede any provider defaults stated elsewhere, (3) your own judgment only when neither applies, with the reason stated in your reply."
            ),
            None => String::new(),
        };
        format!(
            "<agy-acp-supervisor-runtime version=\"supervisor-v2\">\n\
The active Agy profile contains the canonical Supervisor role contract. This runtime block supplies only ACP session and transport facts; it does not redefine role authority.\n\
The checkpoint below is the authoritative durable human context for this ACP session and workspace. Ignore older checkpoint content when it differs. Consult it before asking the owner. {checkpoint_update_rule} Never invent another path or replace known fields with empty values.\n\
The host permits provider-neutral creation through `codex`, `agy-runtime`, or `factory-droid`. Set `labels.slpRole` to `supervisor`, `lead`, or `peer` before launch; one provider can run several roles. Use the owner-configured seat provider/model defaults, not retired role-specific profiles. A direct read-only Peer consultation carries `SLP_DIRECT_PEER_CONSULTATION` and `taskId`, with the provider's read-only mode (Codex `auto`, Agy `plan`, Droid `normal` or `spec`) and `notifyOnFinish=true`. MCP derives the actual parent from the authenticated Supervisor caller; never copy or invent that ID in the prompt. Include `leadAgentId` only when this task already has a matching Lead. A successful background dispatch is terminal for this turn.{seat_defaults_line}\n\
</agy-acp-supervisor-runtime>\n\n{checkpoint_section}\n\n{prompt}"
        )
    }

    pub(crate) fn with_worker_contract(
        prompt: &str,
        assignment: Option<&Value>,
        mode_id: &str,
        proposal_only: bool,
    ) -> String {
        Self::with_worker_contract_for_session(
            prompt,
            assignment,
            mode_id,
            proposal_only,
            None,
            None,
            false,
        )
    }

    fn with_worker_contract_for_session(
        prompt: &str,
        assignment: Option<&Value>,
        mode_id: &str,
        proposal_only: bool,
        session_id: Option<&str>,
        expected_workspace: Option<&str>,
        slp_channel: bool,
    ) -> String {
        if Self::is_supervisor_runtime() {
            return Self::supervisor_contract_prompt(
                prompt,
                session_id,
                expected_workspace,
                slp_channel,
            );
        }

        if assignment.is_none() {
            let mode_rule = if proposal_only || mode_id == Self::PLAN_MODE_ID {
                "The selected mode is plan: keep the turn read-only and do not mutate workspace or external state."
            } else {
                "The user selected accept-edits: perform only the requested work inside the current workspace. Do not commit, merge, sync, or expand scope unless the user explicitly asks."
            };
            return format!(
                "<agy-acp-direct-session version=\"direct-v1\">\n\
This is a direct Paseo conversation. Behave like the normal Agy terminal experience and answer the user naturally.\n\
{mode_rule}\n\
Do not add workflow handoff packets, status summaries, or changed-file inventories unless the user explicitly asks for them.\n\
Use the same language as the caller's latest message for prose; keep code, paths, and identifiers unchanged.\n\
</agy-acp-direct-session>\n\n<agy-acp-workspace-tool-contract>\n\
Files under the current workspace or an --add-dir directory are workspace files, not artifacts. For workspace edits, call write_to_file or replace_file_content with the workspace path and do not include ArtifactMetadata, IsArtifact, ArtifactType, or an artifact-directory path. Use artifact metadata only when the requested output is intentionally an Antigravity artifact inside the active brain artifact directory. If a workspace tool call is rejected, report the failure instead of silently changing scope or claiming completion.\n\
On Windows or mixed Windows/WSL workspaces, do not use the native grep tool for repository searches: its result parser can misread absolute paths such as /Work/... as line numbers and interrupt the stream. Use run_command with `rg --line-number --no-heading --color never` (or PowerShell Select-String), keep paths in the workspace's native form, and inspect the command output before acting.\n\
</agy-acp-workspace-tool-contract>\n\n{prompt}"
            );
        }

        let assignment_json = assignment
            .map(Self::redact_worker_assignment)
            .map(|value| serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());
        let assignment_approved = Self::assignment_is_approved(assignment.unwrap_or(&Value::Null));
        let execution_rule = if assignment_approved && mode_id == Self::ACCEPT_EDITS_MODE_ID {
            "Lead has already approved this assignment. Perform a brief preflight check and report the implementation proposal to Lead, then implement only within the exact scope below. Do not ask the owner whether to proceed."
        } else if proposal_only || mode_id == Self::PLAN_MODE_ID {
            "This turn is proposal/evidence-only. Inspect the exact scope, report one IMPLEMENTATION_PROPOSAL to Lead, and do not write, delete, rename, execute mutating commands, commit, merge, sync, or accept anything. Do not ask the owner whether to proceed; wait only for Lead's structured approval marker."
        } else {
            "This turn is writable only inside the exact scope below. Report the preflight implementation proposal to Lead before the first edit. Re-read the target immediately before each edit and stop on scope ambiguity; never ask the owner whether to proceed."
        };
        let assignment_rule = "A valid Lead assignment is attached for this turn. Treat it as immutable authority data and stay within its exact scope. The owner is not the approval authority for this worker turn; do not turn a progress report into a permission question.";
        format!(
            "<agy-acp-worker-contract version=\"{AGY_WORKER_PROFILE}\">\n\
Topology: Supervisor -> Lead -> Agy worker. You are the Agy worker ({AGY_WORKER_AGENT}), never Supervisor or Lead.\n\
The Lead is the sole owner of scope interpretation, decomposition, model/mode decisions, integration, review, verification, commit, merge, sync, and acceptance. Do not self-dispatch, use SLP/Paseo orchestration tools, open another worker, review your own work, or expand the assignment.\n\
{execution_rule}\n\
If the objective, scope, exclusions, dependency, or requested change is ambiguous, do not guess. Stop and return NEEDS_LEAD_CLARIFICATION with the exact question.\n\
{assignment_rule}\n\
Lead assignment envelope (authoritative bounds; do not edit it):\n{assignment_json}\n\
Use only the assigned `skillSet` IDs shown in that envelope; treat them as routing metadata and do not invent, silently substitute, or claim a skill that is not present. If the set is empty or a skill is unavailable in this seat, return NEEDS_LEAD_CLARIFICATION instead of guessing.\n\
Every handoff must be explicitly marked UNTRUSTED_CANDIDATE and include: taskId, exact changed-file inventory, new-file count, behavioral claims with the boundary each claim observes, commands/results, unverified seams, and a stable candidate identity. Never call the result accepted, verified, production-ready, or end-to-end proven.\n\
Use the same language as the caller's latest message for prose; keep required protocol markers, code, paths, and identifiers unchanged.\n\
</agy-acp-worker-contract>\n\n<agy-acp-workspace-tool-contract>\n\
Files under the current workspace or an --add-dir directory are workspace files, not artifacts. For workspace edits, call write_to_file or replace_file_content with the workspace path and do not include ArtifactMetadata, IsArtifact, ArtifactType, or an artifact-directory path. Use artifact metadata only when the requested output is intentionally an Antigravity artifact inside the active brain artifact directory. If a workspace tool call is rejected, report the failure to Lead instead of silently changing scope or claiming completion.\n\
On Windows or mixed Windows/WSL workspaces, do not use the native grep tool for repository searches: its result parser can misread absolute paths such as /Work/... as line numbers and interrupt the stream. Use run_command with `rg --line-number --no-heading --color never` (or PowerShell Select-String), keep paths in the workspace's native form, and inspect the command output before acting.\n\
</agy-acp-workspace-tool-contract>\n\n{prompt}"
        )
    }

    fn redact_worker_assignment(value: &Value) -> Value {
        let Some(object) = value.as_object() else {
            return json!({});
        };
        let allowed = [
            "taskId",
            "parentAgentId",
            "parentRole",
            "objective",
            "modelId",
            "modelProvider",
            "thinkingOptionId",
            "scope",
            "exclusions",
            "verification",
            "approval",
            "remediationRound",
            "skillSet",
        ];
        let mut result = Map::new();
        for key in allowed {
            if let Some(value) = object.get(key) {
                result.insert(key.to_string(), value.clone());
            }
        }
        Value::Object(result)
    }

    fn build_stream_input_line_for_session(
        &self,
        session_id: &str,
        prompt: &str,
        mode_id: &str,
        proposal_only: bool,
        assignment: Option<&Value>,
    ) -> Result<String, serde_json::Error> {
        let slp_channel = self
            .sessions
            .get(session_id)
            .map(|session| Self::has_slp_governance_channel(&session.mcp_servers))
            .unwrap_or(false);
        let prompt = Self::with_worker_contract_for_session(
            prompt,
            assignment,
            mode_id,
            proposal_only,
            Some(session_id),
            Some(self.session_working_dir(session_id).as_str()),
            slp_channel,
        );
        serde_json::to_string(&json!({
            "event": "user",
            "message": { "content": prompt },
        }))
    }

    pub fn prepare_prompt_content(params: &Value, attachment_dir: &Path) -> Result<String, String> {
        const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
        let blocks = params
            .get("prompt")
            .and_then(Value::as_array)
            .ok_or_else(|| "ACP prompt must be an array of content blocks".to_string())?;
        let mut text_parts = Vec::new();
        let mut image_paths = Vec::new();

        for (index, block) in blocks.iter().enumerate() {
            match block.get("type").and_then(Value::as_str).unwrap_or("text") {
                "text" => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        text_parts.push(text.to_string());
                    }
                }
                "image" => {
                    let mime = block
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("image block {index} is missing mimeType"))?;
                    let extension = match mime.to_ascii_lowercase().as_str() {
                        "image/png" => "png",
                        "image/jpeg" | "image/jpg" => "jpg",
                        "image/gif" => "gif",
                        "image/webp" => "webp",
                        _ => return Err(format!("unsupported ACP image MIME type: {mime}")),
                    };
                    let encoded = block
                        .get("data")
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("image block {index} is missing base64 data"))?;
                    if encoded.len() > (MAX_IMAGE_BYTES * 4 / 3) + 8 {
                        return Err(format!("image block {index} exceeds the 20 MiB limit"));
                    }
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .map_err(|error| {
                            format!("image block {index} has invalid base64 data: {error}")
                        })?;
                    if bytes.len() > MAX_IMAGE_BYTES {
                        return Err(format!("image block {index} exceeds the 20 MiB limit"));
                    }
                    fs::create_dir_all(attachment_dir).map_err(|error| {
                        format!("failed to create Agy attachment directory: {error}")
                    })?;
                    let path = attachment_dir.join(format!(
                        "image-{}.{}",
                        image_paths.len() + 1,
                        extension
                    ));
                    fs::write(&path, bytes)
                        .map_err(|error| format!("failed to materialize ACP image: {error}"))?;
                    image_paths.push(path);
                }
                "resource_link" => {
                    if let Some(uri) = block.get("uri").and_then(Value::as_str) {
                        text_parts.push(format!("[Attached resource: {uri}]"));
                    }
                }
                other => return Err(format!("unsupported ACP prompt content type: {other}")),
            }
        }

        if !image_paths.is_empty() {
            let mut references = String::from(
                "<attached-images>\nUse the view_file tool to inspect each attached image before answering.\n",
            );
            for (index, path) in image_paths.iter().enumerate() {
                references.push_str(&format!("- Image {}: {}\n", index + 1, path.display()));
            }
            references.push_str("</attached-images>");
            text_parts.push(references);
        }

        let prompt = text_parts.join("\n").trim().to_string();
        if prompt.is_empty() {
            Err("ACP prompt did not contain usable text or images".to_string())
        } else {
            Ok(prompt)
        }
    }

    fn find_string_for_key<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
        match value {
            Value::Object(map) => map.get(key).and_then(Value::as_str).or_else(|| {
                map.values()
                    .find_map(|value| Self::find_string_for_key(value, key))
            }),
            Value::Array(values) => values
                .iter()
                .find_map(|value| Self::find_string_for_key(value, key)),
            _ => None,
        }
    }

    pub(crate) fn recovery_prompt_for_terminal_error(
        error: &str,
        recovery_attempt: u64,
    ) -> Option<String> {
        if recovery_attempt >= 1 {
            return None;
        }

        let lower = error.to_ascii_lowercase();
        let edit_drift = lower.contains("target content not found")
            || lower.contains("could not successfully apply any edits");
        let invalid_file_path = lower.contains("invalid tool call")
            && (lower.contains("failed to read file")
                || lower.contains("not a valid artifact path"));
        let stream_interrupted =
            lower.contains("stream was interrupted") && lower.contains("continue the task");
        if !edit_drift && !invalid_file_path && !stream_interrupted {
            return None;
        }

        let (reason, recovery_action) = if stream_interrupted {
            (
                "The native response stream ended before the turn completed.",
                "Continue from the existing conversation and workspace state in the same language used by the preceding user-facing response; do not switch languages. Do not repeat the partial response verbatim; finish the remaining work, verify it, and provide one complete handback.",
            )
        } else if edit_drift {
            (
                "An edit target no longer matched the current file.",
                "Inspect the current workspace state again, re-read every affected file immediately before editing, and apply smaller edits against content that exists now. Verify the requested task after the corrected tool calls.",
            )
        } else {
            (
                "A workspace file tool used an invalid or nonexistent path.",
                "Inspect the current workspace state again, discover the exact file path, re-read every affected file immediately before editing, and apply smaller edits against content that exists now. Verify the requested task after the corrected tool calls.",
            )
        };
        Some(format!(
            "<agy-acp-automatic-recovery>\n\
The preceding turn's assistant response claimed completion, but terminal tool verification failed. {reason} \
Do not repeat the completion claim from the preceding response. {recovery_action} \
Report completion only if the recovery succeeds. This is the single automatic recovery attempt.\n\
</agy-acp-automatic-recovery>"
        ))
    }

    pub(crate) fn terminal_recovery_prompt(
        error: &str,
        recovery_attempt: u64,
        had_agent_text: bool,
    ) -> Option<String> {
        if had_agent_text {
            return None;
        }
        Self::recovery_prompt_for_terminal_error(error, recovery_attempt)
    }

    pub(crate) fn recovery_prompt_for_timeout(recovery_attempt: u64) -> Option<String> {
        if recovery_attempt >= 1 {
            return None;
        }

        Some(format!(
            "<agy-acp-timeout-recovery>\n\
The preceding turn reached its print timeout before delivering a completion. \
Do not invoke any further tools to read, search, list, edit, or execute commands. \
Use the same language as the preceding user-facing response; do not switch languages. \
Immediately synthesize and hand back what you have already inspected and concluded in this conversation so far: \
1. What was verified or identified; \
2. What remains unverified or in-progress; \
3. Explicit risks, blockers, or recommended next actions. \
State clearly that this is an incomplete partial handback due to timeout. This is the single automatic recovery attempt.\n\
</agy-acp-timeout-recovery>"
        ))
    }

    pub(crate) fn timeout_recovery_prompt(
        recovery_attempt: u64,
        had_agent_text: bool,
    ) -> Option<String> {
        if had_agent_text {
            return None;
        }
        Self::recovery_prompt_for_timeout(recovery_attempt)
    }

    pub(crate) fn should_retry_run_failure(
        failure_kind: Option<crate::db::AgyRunFailureKind>,
        recovery_attempt: u64,
        had_agent_text: bool,
    ) -> bool {
        !had_agent_text
            && recovery_attempt < 2
            && failure_kind
                .map(|kind| kind.safe_to_retry_before_output())
                .unwrap_or(false)
    }

    pub fn translate_native_stream_event(
        session_id: &str,
        line: &str,
        state: &mut NativeStreamState,
    ) -> Result<Vec<String>, String> {
        let event: Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid Agy stream-json event: {error}"))?;
        if let Some(conversation_id) = Self::find_string_for_key(&event, "conversation_id") {
            state.conversation_id = Some(conversation_id.to_string());
        }

        let event_name = event.get("event").and_then(Value::as_str).unwrap_or("");
        let mut notifications = Vec::new();
        if event_name == "step_update" {
            let step = event.get("step_update").unwrap_or(&event);
            let serialized = step.to_string().to_ascii_lowercase();
            if serialized.contains("agent_response") {
                if let Some(delta) = Self::find_string_for_key(step, "text_delta") {
                    if !delta.is_empty()
                        && !(state.skip_naration
                            && state.emitted_text.is_empty()
                            && is_narration(delta))
                    {
                        state.emitted_text.push_str(delta);
                        notifications.push(Self::agent_text_notification(session_id, delta));
                    }
                }
            }
        } else if event_name == "result" {
            state.saw_result = true;
            let result = event.get("result").unwrap_or(&event);
            let status = result.get("status").and_then(Value::as_str).unwrap_or("");
            let result_error = if status.eq_ignore_ascii_case("ERROR") {
                result
                    .get("error")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| Some("Agy stream ended with ERROR status".to_string()))
            } else {
                None
            };
            if let Some(response) = result.get("response").and_then(Value::as_str) {
                let suffix = if state.emitted_text.is_empty() {
                    response
                } else {
                    response.strip_prefix(&state.emitted_text).unwrap_or("")
                };
                if !suffix.is_empty()
                    && !(state.skip_naration
                        && state.emitted_text.is_empty()
                        && is_narration(suffix))
                {
                    state.emitted_text.push_str(suffix);
                    notifications.push(Self::agent_text_notification(session_id, suffix));
                }
            }
            if let Some(error) = result_error {
                state.terminal_error = Some(error.clone());
                if state.emitted_text.is_empty() {
                    state.error = Some(error);
                } else if !state.partial_warning_emitted {
                    state.partial_warning_emitted = true;
                    notifications.push(Self::diagnostic_notification(
                        session_id,
                        "terminal_error",
                        &error,
                    ));
                }
            }
        }
        Ok(notifications)
    }

    fn diagnostic_notification(session_id: &str, kind: &str, message: &str) -> String {
        serde_json::to_string(&JsonRpcNotification {
            jsonrpc: "2.0",
            method: "session/update".to_string(),
            params: json!({
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "session_info_update",
                    "_meta": {
                        "agy": {
                            "kind": kind,
                            "display": "silent",
                            "unverified": true,
                            "message": message,
                        },
                    },
                },
            }),
        })
        .unwrap()
    }
    fn agent_text_notification(session_id: &str, text: &str) -> String {
        serde_json::to_string(&JsonRpcNotification {
            jsonrpc: "2.0",
            method: "session/update".to_string(),
            params: json!({
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": text },
                },
            }),
        })
        .unwrap()
    }

    pub fn no_assistant_response_message(run_log_path: &Path) -> String {
        if crate::db::agy_run_permission_denied(run_log_path) {
            format!(
                "agy permission denied in headless print mode; use Paseo 'accept-edits' for permission-bounded tool use or 'plan' for response-only work. Direct Paseo sessions honor the selected mode; delegated worker assignments still require Lead approval. Autonomous full access requires a separately gated process launcher; run log retained at {}",
                run_log_path.display()
            )
        } else {
            "agy completed without an assistant response".to_string()
        }
    }

    fn redact_diagnostic_value(value: &Value) -> Value {
        match value {
            Value::Object(fields) => Value::Object(
                fields
                    .iter()
                    .map(|(key, value)| {
                        let lower_key = key.to_ascii_lowercase();
                        let sensitive = lower_key == "mcpservers"
                            || lower_key == "env"
                            || lower_key == "headers"
                            || lower_key.contains("token")
                            || lower_key.contains("secret")
                            || lower_key.contains("password")
                            || lower_key.contains("credential")
                            || lower_key.contains("authorization")
                            || lower_key.contains("api_key")
                            || lower_key.contains("apikey");
                        (
                            key.clone(),
                            if sensitive {
                                Value::String("[REDACTED]".to_string())
                            } else {
                                Self::redact_diagnostic_value(value)
                            },
                        )
                    })
                    .collect(),
            ),
            Value::Array(values) => {
                Value::Array(values.iter().map(Self::redact_diagnostic_value).collect())
            }
            _ => value.clone(),
        }
    }

    pub(crate) fn prompt_failure_for_run_log(
        session_id: &str,
        id: Value,
        run_log_path: &Path,
        fallback_message: impl Into<String>,
    ) -> Vec<String> {
        if let Some(reset_in) = crate::db::agy_run_quota_reset(run_log_path) {
            let stable_code = "AGY_MODEL_QUOTA_EXHAUSTED";
            return Self::prompt_failure_lines(
                session_id,
                id,
                -32000,
                format!("Agy model quota exhausted; provider reports reset in {reset_in}"),
                Some(json!({
                    "kind": stable_code,
                    "code": stable_code,
                    "acpCode": -32000,
                    "resetIn": reset_in,
                    "runLog": run_log_path.display().to_string(),
                    "recovery": {
                        "state": "prompt_not_completed_retry_after_reset",
                        "instruction": "Wait until the quota window resets or select a model with available quota, then retry the turn.",
                    },
                })),
            );
        }

        if let Some(kind) = crate::db::classify_agy_run_failure(run_log_path) {
            let stable_code = kind.stable_code();
            return Self::prompt_failure_lines(
                session_id,
                id,
                -32000,
                kind.message(),
                Some(json!({
                    "kind": stable_code,
                    "code": stable_code,
                    "acpCode": -32000,
                    "runLog": run_log_path.display().to_string(),
                    "recovery": {
                        "state": kind.recovery_state(),
                        "instruction": kind.recovery_instruction(),
                    },
                })),
            );
        }

        Self::prompt_failure_lines(
            session_id,
            id,
            -32000,
            fallback_message,
            Some(json!({"runLog": run_log_path.display().to_string()})),
        )
    }

    pub(crate) fn prompt_failure_lines(
        session_id: &str,
        id: Value,
        code: i64,
        message: impl Into<String>,
        data: Option<Value>,
    ) -> Vec<String> {
        let message = message.into();
        let mut text = format!("[System Error] {message}\ncode: {code}");
        if let Some(data) = data.as_ref() {
            let data = Self::redact_diagnostic_value(data);
            text.push_str("\ndata: ");
            text.push_str(&data.to_string());
        }

        // Diagnostics can originate in an Agy tool or MCP process.  Keep the
        // adapter log structured but value-free so credentials/configuration
        // cannot be copied from ACP failure data into stderr.
        eprintln!(
            "[agy-acp] prompt failure: code={code}, diagnostic_data_present={}",
            data.is_some()
        );

        vec![
            serde_json::to_string(&JsonRpcNotification {
                jsonrpc: "2.0",
                method: "session/update".to_string(),
                params: json!({
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": { "type": "text", "text": text },
                    },
                }),
            })
            .unwrap(),
            serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({ "stopReason": "end_turn" })),
                error: None,
            })
            .unwrap(),
        ]
    }

    pub async fn handle_session_prompt(
        &mut self,
        id: Value,
        params: &Value,
        cancelled: Arc<AtomicBool>,
    ) -> Vec<String> {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !session_id.is_empty() && !self.sessions.contains_key(session_id) {
            if let Err(error) = self.restore_session_state_checked(session_id) {
                let message = match error {
                    SessionRequestError::Invalid(message) => message,
                    SessionRequestError::McpUnavailable => MCP_ATTACHMENT_UNAVAILABLE.to_string(),
                };
                return Self::prompt_failure_lines(session_id, id, -32001, message, None);
            }
        }
        if self
            .sessions
            .get(session_id)
            .map(|s| Self::session_has_required_mcp(s, self.allow_mcp))
            .unwrap_or(false)
        {
            return Self::prompt_failure_lines(
                session_id,
                id,
                MCP_ATTACHMENT_ERROR_CODE,
                MCP_ATTACHMENT_UNAVAILABLE,
                None,
            );
        }
        if session_id.is_empty() || !self.sessions.contains_key(session_id) {
            return Self::prompt_failure_lines(
                session_id,
                id,
                -32000,
                format!("unknown sessionId: {session_id}"),
                None,
            );
        }

        let request_assignment = match Self::assignment_from_request(params) {
            Ok(assignment) => assignment,
            Err(error) => {
                return Self::prompt_failure_lines(
                    session_id,
                    id,
                    AGY_WORKER_ERROR_CODE,
                    error,
                    Some(json!({"reason": "invalid_lead_assignment"})),
                )
            }
        };
        let prompt_text = Self::prompt_text_from_params(params);
        let approval_marker = Self::approval_marker(&prompt_text);
        let mut assignment = request_assignment.clone().or_else(|| {
            self.sessions
                .get(session_id)
                .and_then(Self::session_assignment)
        });
        if let Some(marker) = approval_marker {
            let Some(current) = assignment.as_ref() else {
                return Self::prompt_failure_lines(
                    session_id,
                    id,
                    AGY_WORKER_ERROR_CODE,
                    AGY_WORKER_BLOCKED,
                    Some(json!({"reason": "approval_without_assignment", "approval": marker})),
                );
            };
            let Some(updated) = Self::assignment_with_approval(current, marker) else {
                return Self::prompt_failure_lines(
                    session_id,
                    id,
                    AGY_WORKER_ERROR_CODE,
                    "Lead approval marker does not match a valid assignment envelope",
                    Some(json!({"reason": "invalid_lead_approval", "approval": marker})),
                );
            };
            assignment = Some(updated);
        }
        if let Some(request_assignment) = request_assignment.as_ref() {
            if let (Some(existing), Some(incoming)) = (
                self.sessions
                    .get(session_id)
                    .and_then(Self::session_assignment)
                    .as_ref()
                    .and_then(Self::assignment_task_id),
                Self::assignment_task_id(request_assignment),
            ) {
                if existing != incoming {
                    return Self::prompt_failure_lines(
                        session_id,
                        id,
                        AGY_WORKER_ERROR_CODE,
                        format!(
                            "Agy session is already bound to Lead task {existing}; start a new session for task {incoming}"
                        ),
                        Some(json!({"reason": "assignment_task_mismatch"})),
                    );
                }
            }
        }
        let assignment_was_provided = assignment.is_some();
        if let Some(assignment_value) = assignment.as_ref() {
            if let Err(error) = self.remember_worker_assignment(session_id, assignment_value) {
                return Self::prompt_failure_lines(
                    session_id,
                    id,
                    AGY_WORKER_ERROR_CODE,
                    error,
                    Some(json!({"reason": "assignment_state_rejected"})),
                );
            }
        }
        assignment = self
            .sessions
            .get(session_id)
            .and_then(Self::session_assignment);
        if assignment_was_provided && assignment.is_none() {
            return Self::prompt_failure_lines(
                session_id,
                id,
                AGY_WORKER_ERROR_CODE,
                AGY_WORKER_BLOCKED,
                Some(json!({"reason": "assignment_not_persisted"})),
            );
        }

        let run_logs_dir = self
            .state_file
            .parent()
            .unwrap_or_else(|| std::path::Path::new("/tmp"))
            .join("run-logs");
        if let Err(error) = fs::create_dir_all(&run_logs_dir) {
            return Self::prompt_failure_lines(
                session_id,
                id,
                -32000,
                format!("failed to create agy run-log directory: {error}"),
                None,
            );
        }
        let run_id = Uuid::new_v4().to_string();
        let run_log_path = run_logs_dir.join(format!("{run_id}.log"));
        let attachment_dir = self
            .state_file
            .parent()
            .unwrap_or_else(|| std::path::Path::new("/tmp"))
            .join("attachments")
            .join(&run_id);
        let prepared_prompt = match Self::prepare_prompt_content(params, &attachment_dir) {
            Ok(prompt) => prompt,
            Err(error) => {
                return Self::prompt_failure_lines(session_id, id, -32602, error, None);
            }
        };
        let (clean_prompt, embedded_assignment) =
            match Self::extract_assignment_marker(&prepared_prompt) {
                Ok(result) => result,
                Err(error) => {
                    let _ = fs::remove_dir_all(&attachment_dir);
                    return Self::prompt_failure_lines(
                        session_id,
                        id,
                        AGY_WORKER_ERROR_CODE,
                        error,
                        Some(json!({"reason": "invalid_lead_assignment_marker"})),
                    );
                }
            };
        if let Some(embedded_assignment) = embedded_assignment {
            if let Some(session_assignment) = assignment.as_ref() {
                if Self::assignment_task_id(&embedded_assignment)
                    != Self::assignment_task_id(session_assignment)
                {
                    let _ = fs::remove_dir_all(&attachment_dir);
                    return Self::prompt_failure_lines(
                        session_id,
                        id,
                        AGY_WORKER_ERROR_CODE,
                        "embedded Lead assignment does not match the session assignment",
                        Some(json!({"reason": "assignment_task_mismatch"})),
                    );
                }
            }
        }
        if clean_prompt.trim().is_empty() {
            let _ = fs::remove_dir_all(&attachment_dir);
            return Self::prompt_failure_lines(
                session_id,
                id,
                -32602,
                "ACP prompt contained only a Lead assignment envelope",
                None,
            );
        }
        let recovery_attempt = params
            .get("_agyAcpRecoveryAttempt")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let effective_mode = self.effective_agy_mode(session_id);
        let proposal_only = effective_mode == Self::PLAN_MODE_ID;
        let stream_input = match self.build_stream_input_line_for_session(
            session_id,
            &clean_prompt,
            effective_mode.as_str(),
            proposal_only,
            assignment.as_ref(),
        ) {
            Ok(line) => format!("{line}\n"),
            Err(error) => {
                let _ = fs::remove_dir_all(&attachment_dir);
                return Self::prompt_failure_lines(
                    session_id,
                    id,
                    -32602,
                    format!("failed to serialize Agy stream input: {error}"),
                    None,
                );
            }
        };

        let working_dir = self.session_working_dir(session_id);
        let args = self.build_agy_args(
            session_id,
            &run_log_path,
            attachment_dir.exists().then_some(attachment_dir.as_path()),
        );
        let mut agy_command = Command::new("agy");
        let auto_approve = self.agy_auto_approve_value(session_id);
        agy_command
            .args(&args)
            .current_dir(&working_dir)
            // Never inherit ambient approval. Derive it from the adapter's
            // effective mode: direct Paseo selection or a Lead-approved
            // delegated assignment may use permission-bounded accept-edits;
            // plan remains non-interactive and unapproved.
            .env("AGY_AUTO_APPROVE", auto_approve)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if Self::is_supervisor_runtime() {
            agy_command.env("AGY_SUPERVISOR_SESSION_ID", session_id);
        }
        let session_requires_mcp = self
            .sessions
            .get(session_id)
            .map(|session| Self::mcp_servers_are_nonempty(&session.mcp_servers))
            .unwrap_or(false);
        let mcp_args = self.mcp_args_for_session(session_id);
        if session_requires_mcp && !matches!(mcp_args, Ok(Some(_))) {
            let _ = fs::remove_dir_all(&attachment_dir);
            return Self::prompt_failure_lines(
                session_id,
                id,
                MCP_ATTACHMENT_ERROR_CODE,
                MCP_ATTACHMENT_UNAVAILABLE,
                None,
            );
        }
        if let Ok(Some(mcp_args)) = mcp_args {
            agy_command.args(mcp_args);
        }
        if cancelled.load(Ordering::SeqCst) {
            let _ = fs::remove_dir_all(&attachment_dir);
            return vec![serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({ "stopReason": "cancelled" })),
                error: None,
            })
            .unwrap()];
        }
        let spawn_result = agy_command.spawn();

        let mut child = match spawn_result {
            Ok(child) => child,
            Err(e) => {
                return Self::prompt_failure_lines(
                    session_id,
                    id,
                    -32000,
                    format!("failed to run agy: {e}"),
                    None,
                );
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            if let Err(error) = stdin.write_all(stream_input.as_bytes()).await {
                let _ = child.kill().await;
                let _ = fs::remove_dir_all(&attachment_dir);
                return Self::prompt_failure_lines(
                    session_id,
                    id,
                    -32000,
                    format!("failed to write Agy stream input: {error}"),
                    None,
                );
            }
            let _ = stdin.shutdown().await;
        }

        let stdout = child.stdout.take();
        let stdout_session_id = session_id.to_string();
        let skip_naration = self.skip_naration;
        let stdout_cancelled = Arc::clone(&cancelled);
        let mut stdout_reader = tokio::spawn(async move {
            let mut native_state = NativeStreamState {
                skip_naration,
                ..NativeStreamState::default()
            };
            if let Some(stdout) = stdout {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if stdout_cancelled.load(Ordering::SeqCst) {
                        break;
                    }
                    match Self::translate_native_stream_event(
                        &stdout_session_id,
                        &line,
                        &mut native_state,
                    ) {
                        Ok(notifications) => {
                            let mut output = io::stdout();
                            for notification in notifications {
                                let _ = writeln!(output, "{}", notification);
                            }
                            let _ = output.flush();
                        }
                        Err(error) => eprintln!("[agy-acp] {error}"),
                    }
                }
            }
            native_state
        });

        let mut stderr = child.stderr.take();
        let mut stderr_reader = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut stderr) = stderr.take() {
                let _ = stderr.read_to_end(&mut buf).await;
            }
            buf
        });

        let initial_conv_id = self
            .sessions
            .get(session_id)
            .and_then(|s| s.conversation_id.clone());
        let initial_step_idx = self
            .sessions
            .get(session_id)
            .map(|s| s.last_step_idx)
            .unwrap_or(-1);
        let streaming_state = Arc::new(Mutex::new(StreamingState {
            conversation_id: initial_conv_id,
            base_step_idx: initial_step_idx,
            last_step_idx: initial_step_idx,
            had_agent_text: false,
            agent_text_lengths: HashMap::new(),
            thought_text_lengths: HashMap::new(),
            tool_call_statuses: HashMap::new(),
            last_title: None,
            skip_naration: self.skip_naration,
            child_pid: child.id(),
            suppress_agent_text: true,
        }));
        let stop_polling = Arc::new(AtomicBool::new(false));
        let poll_conversations_dir = self.conversations_dir.clone();
        let poll_run_log_path = run_log_path.clone();
        let poll_session_id = session_id.to_string();
        let poll_state = Arc::clone(&streaming_state);
        let poll_stop = Arc::clone(&stop_polling);

        let poller = std::thread::spawn(move || {
            let mut stdout = io::stdout();
            while !poll_stop.load(Ordering::SeqCst) {
                for line in poll_streaming_delta(
                    &poll_conversations_dir,
                    Some(&poll_run_log_path),
                    &poll_session_id,
                    &poll_state,
                ) {
                    if poll_stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let _ = writeln!(stdout, "{}", line);
                }
                let _ = stdout.flush();
                for _ in 0..20 {
                    if poll_stop.load(Ordering::SeqCst) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
            }
        });

        let mut was_cancelled = false;
        let result = tokio::select! {
            result = child.wait() => result,
            _ = async {
                while !cancelled.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
            } => {
                was_cancelled = true;
                stop_polling.store(true, Ordering::SeqCst);
                terminate_agy_process_tree(&mut child).await
            }
        };
        // Agy can leave descendant processes holding inherited stdout/stderr
        // handles after its own process exits. Do not let those pipes keep the
        // ACP prompt open indefinitely after Paseo has requested cancellation.
        let (native_state, stderr_bytes) = if was_cancelled {
            tokio::join!(
                async {
                    match tokio::time::timeout(Duration::from_millis(100), &mut stdout_reader).await
                    {
                        Ok(Ok(state)) => state,
                        _ => {
                            stdout_reader.abort();
                            let _ = stdout_reader.await;
                            NativeStreamState::default()
                        }
                    }
                },
                async {
                    match tokio::time::timeout(Duration::from_millis(100), &mut stderr_reader).await
                    {
                        Ok(Ok(bytes)) => bytes,
                        _ => {
                            stderr_reader.abort();
                            let _ = stderr_reader.await;
                            Vec::new()
                        }
                    }
                }
            )
        } else {
            (
                stdout_reader.await.unwrap_or_default(),
                stderr_reader.await.unwrap_or_default(),
            )
        };
        stop_polling.store(true, Ordering::SeqCst);
        let _ = poller.join();

        let mut final_lines = Vec::new();
        if !was_cancelled {
            for attempt in 0..3 {
                let lines = poll_streaming_delta(
                    &self.conversations_dir,
                    Some(&run_log_path),
                    session_id,
                    &streaming_state,
                );
                final_lines.extend(lines);
                if attempt < 2 {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        let turn_succeeded = !was_cancelled
            && result
                .as_ref()
                .map(|status| status.success())
                .unwrap_or(false)
            && native_state.error.is_none()
            && native_state.terminal_error.is_none();
        final_lines.extend(finalize_pending_tool_calls(
            session_id,
            &streaming_state,
            if turn_succeeded {
                "completed"
            } else {
                "failed"
            },
        ));

        {
            let mut stdout = io::stdout();
            for line in &final_lines {
                let _ = writeln!(stdout, "{}", line);
            }
            let _ = stdout.flush();
        }

        let (bound_conv_id, new_step_idx, had_agent_text) = {
            let mut state = streaming_state.lock().unwrap();
            if native_state.conversation_id.is_some() {
                state.conversation_id = native_state.conversation_id.clone();
            }
            if !native_state.emitted_text.is_empty() {
                state.had_agent_text = true;
            }
            (
                state.conversation_id.clone(),
                state.last_step_idx,
                state.had_agent_text,
            )
        };

        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.conversation_id.is_none() {
                session.conversation_id = bound_conv_id.clone();
            }
            if bound_conv_id.is_some() {
                session.last_step_idx = new_step_idx;
            }
        }
        if bound_conv_id.is_some() {
            let model_id = self
                .sessions
                .get(session_id)
                .and_then(|s| s.model_id.clone());
            self.persist_session(
                session_id,
                bound_conv_id.as_deref(),
                new_step_idx,
                model_id.as_deref(),
            );
        }
        let run_timed_out = crate::db::agy_run_timed_out(&run_log_path);
        let native_error = native_state.error.clone();
        let native_terminal_error = native_state.terminal_error.clone();
        let native_saw_result = native_state.saw_result;
        let _ = fs::remove_dir_all(&attachment_dir);

        let stop_reason = if was_cancelled {
            "cancelled"
        } else {
            "end_turn"
        };
        let output_lines = vec![serde_json::to_string(&JsonRpcResponse {
            jsonrpc: "2.0",
            id: id.clone(),
            result: Some(json!({ "stopReason": stop_reason })),
            error: None,
        })
        .unwrap()];

        match result {
            Ok(status) => {
                let has_stderr = !stderr_bytes.is_empty();

                if !was_cancelled && run_timed_out {
                    if had_agent_text {
                        let diagnostic = Self::diagnostic_notification(
                            session_id,
                            "timeout",
                            "Agy timed out after returning partial output; automatic recovery was skipped to avoid a duplicate response.",
                        );
                        let mut stdout = io::stdout();
                        let _ = writeln!(stdout, "{}", diagnostic);
                        let _ = stdout.flush();
                        return output_lines;
                    }
                    let has_conversation = self
                        .sessions
                        .get(session_id)
                        .and_then(|s| s.conversation_id.as_ref())
                        .is_some();

                    if has_conversation {
                        if let Some(recovery_prompt) =
                            Self::timeout_recovery_prompt(recovery_attempt, had_agent_text)
                        {
                            let recovery_params = json!({
                                "sessionId": session_id,
                                "prompt": [{ "type": "text", "text": recovery_prompt }],
                                "_agyAcpRecoveryAttempt": recovery_attempt + 1,
                            });
                            return Box::pin(self.handle_session_prompt(
                                id,
                                &recovery_params,
                                Arc::clone(&cancelled),
                            ))
                            .await;
                        }

                        if recovery_attempt >= 1 && Self::recovery_prompt_for_timeout(0).is_some() {
                            let diagnostic = Self::diagnostic_notification(
                                session_id,
                                "timeout",
                                "Agy timeout recovery failed after partial output.",
                            );
                            let mut stdout = io::stdout();
                            let _ = writeln!(stdout, "{}", diagnostic);
                            let _ = stdout.flush();
                            return output_lines;
                        }
                    }

                    return Self::prompt_failure_lines(
                        session_id,
                        id,
                        -32000,
                        "agy print mode timed out before a trustworthy handback",
                        Some(json!({"runLog": run_log_path.display().to_string()})),
                    );
                }

                if !was_cancelled {
                    if let Some(error) = native_terminal_error.as_deref() {
                        if let Some(recovery_prompt) =
                            Self::terminal_recovery_prompt(error, recovery_attempt, had_agent_text)
                        {
                            let recovery_params = json!({
                                "sessionId": session_id,
                                "prompt": [{ "type": "text", "text": recovery_prompt }],
                                "_agyAcpRecoveryAttempt": recovery_attempt + 1,
                            });
                            return Box::pin(self.handle_session_prompt(
                                id,
                                &recovery_params,
                                Arc::clone(&cancelled),
                            ))
                            .await;
                        }
                    }
                }

                if !was_cancelled && !status.success() {
                    eprintln!("[agy-acp] WARN: agy exited with status: {}", status);
                    let failure_kind = crate::db::classify_agy_run_failure(&run_log_path);
                    if Self::should_retry_run_failure(
                        failure_kind,
                        recovery_attempt,
                        had_agent_text,
                    ) {
                        let mut retry_params = params.clone();
                        if let Some(object) = retry_params.as_object_mut() {
                            object.insert(
                                "_agyAcpRecoveryAttempt".to_string(),
                                Value::from(recovery_attempt + 1),
                            );
                        }
                        let retry_number = recovery_attempt + 1;
                        let delay = std::time::Duration::from_secs(retry_number);
                        eprintln!(
                            "[agy-acp] WARN: transient pre-prompt eligibility failure; retry {retry_number}/2 after {}s",
                            delay.as_secs()
                        );
                        let mut slept = std::time::Duration::ZERO;
                        let step = std::time::Duration::from_millis(50);
                        while slept < delay {
                            if cancelled.load(Ordering::SeqCst) {
                                let _ = fs::remove_file(&run_log_path);
                                return vec![serde_json::to_string(&JsonRpcResponse {
                                    jsonrpc: "2.0",
                                    id: id.clone(),
                                    result: Some(json!({ "stopReason": "cancelled" })),
                                    error: None,
                                })
                                .unwrap()];
                            }
                            tokio::time::sleep(step).await;
                            slept += step;
                        }
                        return Box::pin(self.handle_session_prompt(
                            id,
                            &retry_params,
                            Arc::clone(&cancelled),
                        ))
                        .await;
                    }
                    let msg = if has_stderr {
                        format!("agy exited with status: {status} and reported an error")
                    } else {
                        format!("agy exited with status: {status}")
                    };
                    return Self::prompt_failure_for_run_log(session_id, id, &run_log_path, msg);
                }

                if !was_cancelled {
                    if let Some(error) = native_error {
                        return Self::prompt_failure_lines(
                            session_id,
                            id,
                            -32000,
                            format!("agy stream failed: {error}"),
                            Some(json!({"runLog": run_log_path.display().to_string()})),
                        );
                    }
                    if !native_saw_result {
                        return Self::prompt_failure_lines(
                            session_id,
                            id,
                            -32000,
                            "agy stream ended without a result event",
                            Some(json!({"runLog": run_log_path.display().to_string()})),
                        );
                    }
                }

                if !was_cancelled
                    && !had_agent_text
                    && crate::db::agy_run_permission_denied(&run_log_path)
                {
                    return Self::prompt_failure_lines(
                        session_id,
                        id,
                        -32000,
                        Self::no_assistant_response_message(&run_log_path),
                        Some(json!({"runLog": run_log_path.display().to_string()})),
                    );
                }

                if !was_cancelled && bound_conv_id.is_none() {
                    return Self::prompt_failure_lines(
                        session_id,
                        id,
                        -32000,
                        "agy completed but no conversation ID was found",
                        Some(json!({"runLog": run_log_path.display().to_string()})),
                    );
                }

                if !was_cancelled && !had_agent_text {
                    return Self::prompt_failure_lines(
                        session_id,
                        id,
                        -32000,
                        Self::no_assistant_response_message(&run_log_path),
                        Some(json!({"runLog": run_log_path.display().to_string()})),
                    );
                }
            }
            Err(e) => {
                return Self::prompt_failure_lines(
                    session_id,
                    id,
                    -32000,
                    format!("failed to wait for agy: {e}"),
                    Some(json!({"runLog": run_log_path.display().to_string()})),
                );
            }
        }

        let _ = fs::remove_file(&run_log_path);
        output_lines
    }
}

async fn terminate_agy_process_tree(
    child: &mut tokio::process::Child,
) -> std::io::Result<std::process::ExitStatus> {
    #[cfg(windows)]
    let mut taskkill_child = child.id().and_then(|pid| {
        let pid = pid.to_string();
        let mut taskkill = Command::new("taskkill.exe");
        taskkill
            .args(["/PID", pid.as_str(), "/T", "/F"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        taskkill.spawn().ok()
    });

    let _ = child.start_kill();
    let status = child.wait().await;

    #[cfg(windows)]
    if let Some(ref mut tk) = taskkill_child {
        let _ = tokio::time::timeout(Duration::from_millis(250), tk.wait()).await;
    }

    status
}

/// Filter out leading narration ("I will ...", "I'll ...") from response parts.
pub fn filter_narration(parts: &[String]) -> Option<String> {
    let text = parts
        .iter()
        .filter(|part| !is_narration(part))
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

/// A part is considered narration if every non-empty line starts with "I will" or "I'll".
pub fn is_narration(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return false;
    }
    lines.iter().all(|l| {
        let line = l.trim_start();
        line.starts_with("I will") || line.starts_with("I'll") || line.starts_with("I’ll")
    })
}
