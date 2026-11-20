use rusqlite::Connection;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use uuid::Uuid;

use crate::adapter::{filter_narration, Adapter, DEFAULT_PRINT_TIMEOUT};
use crate::protobuf::{
    extract_text_from_step_payload, extract_thought_from_step_payload,
    extract_title_from_step_payload, extract_tool_name, extract_tool_update_from_step_payload,
    extract_user_text_from_step_payload, read_varint,
};
use crate::streaming::{finalize_pending_tool_calls, poll_streaming_delta};
use crate::types::StreamingState;
use crate::Cli;
use clap::Parser;

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        if value < 128 {
            out.push(value as u8);
            break;
        }
        out.push(((value as u8) & 0x7F) | 0x80);
        value >>= 7;
    }
}

fn push_len_field(out: &mut Vec<u8>, field_number: u64, bytes: &[u8]) {
    push_varint(out, (field_number << 3) | 2);
    push_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn push_varint_field(out: &mut Vec<u8>, field_number: u64, value: u64) {
    push_varint(out, field_number << 3);
    push_varint(out, value);
}

fn make_assistant_payload(text: &str) -> Vec<u8> {
    let mut inner = Vec::new();
    push_len_field(&mut inner, 1, text.as_bytes());

    let mut outer = Vec::new();
    push_len_field(&mut outer, 20, &inner);
    outer
}

#[test]
fn test_parse_skip_naration_flag() {
    let legacy = Cli::try_parse_from(["agy-acp", "--skip-naration"]).unwrap();
    assert!(legacy.skip_naration);
    assert!(!legacy.effective_skip_naration());
    assert!(!Cli::try_parse_from(["agy-acp"]).unwrap().skip_naration);
    assert!(Cli::try_parse_from(["agy-acp", "--skip-narration"]).is_err());
    assert!(Cli::try_parse_from(["agy-acp", "--allow-autonomous-root-full-access"]).is_err());
    assert!(
        !Cli::try_parse_from(["agy-acp"])
            .unwrap()
            .allow_lead_full_access
    );
    assert!(
        Cli::try_parse_from(["agy-acp", "--allow-lead-full-access"])
            .unwrap()
            .allow_lead_full_access
    );
    assert!(!Cli::try_parse_from(["agy-acp"]).unwrap().auto_approve);
    assert!(
        Cli::try_parse_from(["agy-acp", "--auto-approve"])
            .unwrap()
            .auto_approve
    );
    assert!(
        Cli::try_parse_from(["agy-acp", "--dangerously-skip-permissions"])
            .unwrap()
            .dangerously_skip_permissions
    );
}

#[test]
fn test_auto_approve_env_and_mode_handling() {
    let root = std::env::temp_dir().join(format!("agy-acp-auto-approve-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut adapter = adapter_for_test_root(&root);
    let session = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = session.result.unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    // Default: auto_approve is false, no --dangerously-skip-permissions
    let args = adapter.build_agy_args(&session_id, &root.join("run.log"), None);
    assert!(!args
        .iter()
        .any(|arg| arg == "--dangerously-skip-permissions"));

    // Ambient/legacy auto_approve cannot bypass the Lead approval gate.
    adapter.auto_approve = true;
    let args_approved = adapter.build_agy_args(&session_id, &root.join("run.log"), None);
    assert!(!args_approved
        .iter()
        .any(|arg| arg == "--dangerously-skip-permissions"));
    let mode_index = args_approved
        .iter()
        .position(|arg| arg == "--mode")
        .unwrap();
    assert_eq!(args_approved[mode_index + 1], "plan");

    // In plan mode: --dangerously-skip-permissions is NOT added even with auto_approve = true
    adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "mode", "value": "plan"}),
    );
    let args_plan = adapter.build_agy_args(&session_id, &root.join("run.log"), None);
    assert!(!args_plan
        .iter()
        .any(|arg| arg == "--dangerously-skip-permissions"));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_default_print_timeout_is_one_hour() {
    assert_eq!(DEFAULT_PRINT_TIMEOUT, "1h");
}

fn make_user_payload(text: &str) -> Vec<u8> {
    let mut content = Vec::new();
    push_len_field(&mut content, 1, text.as_bytes());

    let mut prompt = Vec::new();
    push_len_field(&mut prompt, 2, text.as_bytes());
    push_len_field(&mut prompt, 3, &content);

    let mut outer = Vec::new();
    push_len_field(&mut outer, 19, &prompt);
    outer
}

fn make_title_payload(title: &str) -> Vec<u8> {
    let mut title_update = Vec::new();
    push_len_field(&mut title_update, 4, title.as_bytes());

    let mut outer = Vec::new();
    push_len_field(&mut outer, 30, &title_update);
    outer
}

fn make_tool_payload(
    call_id: &str,
    tool_name: &str,
    input_json: &str,
    summary: &str,
    result_field: Option<(u64, Vec<u8>)>,
) -> Vec<u8> {
    let mut call = Vec::new();
    push_len_field(&mut call, 1, call_id.as_bytes());
    push_len_field(&mut call, 2, tool_name.as_bytes());
    push_len_field(&mut call, 3, input_json.as_bytes());
    push_len_field(&mut call, 9, tool_name.as_bytes());

    let mut tool = Vec::new();
    push_len_field(&mut tool, 4, &call);
    push_len_field(&mut tool, 30, summary.as_bytes());

    let mut outer = Vec::new();
    push_varint_field(&mut outer, 1, 7);
    push_len_field(&mut outer, 5, &tool);
    if let Some((field, result)) = result_field {
        push_len_field(&mut outer, field, &result);
    }
    outer
}

fn adapter_for_test_root(root: &std::path::Path) -> Adapter {
    Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec!["test-model".to_string()],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    }
}

fn adapter_for_test() -> Adapter {
    let root = std::env::temp_dir().join(format!("agy-acp-unit-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    adapter_for_test_root(&root)
}

#[test]
fn test_session_scoped_mcp_overrides_unscoped_global_config() {
    let root = std::env::temp_dir().join(format!("agy-acp-scoped-mcp-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let upstream = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let upstream_addr = upstream.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let upstream_thread = std::thread::spawn(move || {
        let (mut stream, _) = upstream.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            let read = stream.read(&mut buffer).unwrap();
            request.extend_from_slice(&buffer[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        request_tx
            .send(String::from_utf8(request).unwrap())
            .unwrap();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
            .unwrap();
    });
    let mut adapter = adapter_for_test_root(&root);
    adapter.allow_mcp = true;
    let response = adapter.handle_session_new(
        json!(1),
        &json!({
            "cwd": "D:\\Work\\Project",
            "mcpServers": [{
                "type": "http",
                "name": "paseo",
                "url": format!("http://{upstream_addr}/mcp/agents?callerAgentId=agy-parent"),
                "headers": [{"name": "Authorization", "value": "Bearer synthetic"}]
            }],
            "modeId": "plan"
        }),
    );
    assert!(response.error.is_none());
    let session_id = response.result.unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let env = adapter
        .mcp_env_for_session(&session_id)
        .expect("accepted MCP context must be attached to the Agy subprocess");
    assert!(
        env.iter().all(|(name, _)| !matches!(
            name.as_str(),
            "HOME" | "USERPROFILE" | "XDG_CONFIG_HOME" | "XDG_DATA_HOME"
        )),
        "scoped MCP must not hide Agy's authenticated conversation store"
    );
    let seat = adapter.mcp_registry.get(&session_id).unwrap();
    let agy_args = adapter.mcp_args_for_session(&session_id).unwrap().unwrap();
    let gemini_dir = PathBuf::from(
        agy_args
            .iter()
            .find_map(|arg| arg.strip_prefix("--gemini_dir="))
            .unwrap(),
    );
    let app_data_relative = PathBuf::from(
        agy_args
            .iter()
            .find_map(|arg| arg.strip_prefix("--app_data_dir="))
            .unwrap(),
    );
    assert!(gemini_dir.is_absolute());
    assert!(!app_data_relative.is_absolute());
    assert_eq!(
        fs::canonicalize(gemini_dir.join(app_data_relative)).unwrap(),
        fs::canonicalize(&root).unwrap(),
        "the scoped config must keep Agy's original app-data/conversation store"
    );
    let config: Value = serde_json::from_slice(&fs::read(seat.config_path()).unwrap()).unwrap();
    let local_url = config["mcpServers"]["paseo"]["serverUrl"].as_str().unwrap();
    assert!(local_url.starts_with("http://127.0.0.1:"));
    assert_ne!(local_url, "http://127.0.0.1:6767/mcp/agents");

    let local_endpoint = local_url.strip_prefix("http://").unwrap();
    let (authority, path) = local_endpoint.split_once('/').unwrap();
    let mut local = TcpStream::connect(authority).unwrap();
    local
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    write!(
        local,
        "POST /{path} HTTP/1.1\r\nHost: {authority}\r\nContent-Length: 2\r\n\r\n{{}}"
    )
    .unwrap();
    let mut response_body = String::new();
    local.read_to_string(&mut response_body).unwrap();
    assert!(response_body.ends_with("OK"));
    let forwarded = request_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(forwarded.starts_with("POST /mcp/agents?callerAgentId=agy-parent HTTP/1.1\r\n"));
    assert!(forwarded.contains("authorization: Bearer synthetic\r\n"));
    upstream_thread.join().unwrap();

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_extract_text_from_step_payload_field20_field1() {
    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(0x05);
    inner.extend_from_slice(b"hello");

    let mut blob = vec![0x08, 0x0F, 0xA2, 0x01, inner.len() as u8];
    blob.extend_from_slice(&inner);
    assert_eq!(
        extract_text_from_step_payload(&blob),
        Some("hello".to_string())
    );
}

#[test]
fn test_extract_text_returns_none_without_field20() {
    let blob = vec![0x08, 0x03];
    assert_eq!(extract_text_from_step_payload(&blob), None);
}

#[test]
fn test_extract_thought_from_step_payload_field20_field3() {
    let mut inner = Vec::new();
    push_len_field(&mut inner, 1, b"I will inspect the script.");
    push_len_field(
        &mut inner,
        3,
        b"**Analyzing Container Startup**\nI need to verify how the container reuse check works.",
    );

    let mut blob = Vec::new();
    push_varint_field(&mut blob, 1, 15);
    push_len_field(&mut blob, 20, &inner);

    assert_eq!(
        extract_thought_from_step_payload(&blob),
        Some(
            "**Analyzing Container Startup**\nI need to verify how the container reuse check works."
                .to_string()
        )
    );
}

#[test]
fn test_extract_user_text_from_step_payload_field19_field2() {
    let payload = make_user_payload("how are you?");
    assert_eq!(
        extract_user_text_from_step_payload(&payload),
        Some("how are you?".to_string())
    );
}

#[test]
fn test_extract_title_from_step_payload_field30_field4() {
    let payload = make_title_payload("Documenting Conversation Snapshot Function");
    assert_eq!(
        extract_title_from_step_payload(&payload),
        Some("Documenting Conversation Snapshot Function".to_string())
    );
}

#[test]
fn test_extract_title_ignores_empty_title() {
    assert_eq!(
        extract_title_from_step_payload(&make_title_payload("  ")),
        None
    );
}

#[test]
fn test_extract_text_multiline() {
    let text = b"Safe memory rules\nCompiler points out the flaws\nFast and fearless code";
    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(text.len() as u8);
    inner.extend_from_slice(text);

    let mut blob = vec![0x08, 0x01, 0xA2, 0x01, inner.len() as u8];
    blob.extend_from_slice(&inner);
    assert_eq!(
        extract_text_from_step_payload(&blob),
        Some(
            "Safe memory rules\nCompiler points out the flaws\nFast and fearless code".to_string()
        )
    );
}

#[test]
fn test_extract_tool_update_from_step_payload_json() {
    let payload = br#"
        grep_search
        {"Query":"prompt","SearchPath":"/tmp/project/src/main.rs","toolAction":"Finding prompt handling","toolSummary":"Grep prompt"}
    "#;

    let update = extract_tool_update_from_step_payload(19, 7, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["toolCallId"], "agy-19-7");
    assert_eq!(update["title"], "Grep prompt");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["status"], "in_progress");
    assert_eq!(update["rawInput"]["Query"], "prompt");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src/main.rs");
}

#[test]
/// Tests that when a tool payload lacks a JSON body (and thus has no `toolSummary`
/// or `toolAction`), the extractor falls back to using the extracted tool name
/// (e.g., `view_file`) as the update title.
fn test_extract_tool_update_uses_tool_name_fallback() {
    let payload = b"view_file";
    let update = extract_tool_update_from_step_payload(3, 8, payload).unwrap();
    assert_eq!(update["title"], "view_file");
    assert_eq!(update["kind"], "read");
}

#[test]
fn test_extract_tool_update_ignores_single_letter_noise() {
    let payload = b"P";
    assert_eq!(extract_tool_update_from_step_payload(4, 17, payload), None);
}

#[test]
fn test_extract_tool_update_ignores_generic_message_fallback() {
    let payload = b"Message";
    assert_eq!(extract_tool_update_from_step_payload(5, 17, payload), None);
}

#[test]
fn test_extract_tool_update_parses_first_balanced_json_object() {
    let payload = br#"
        abc123 view_file
        {"AbsolutePath":"/tmp/project/README.md","toolAction":"Reading README.md","toolSummary":"View README file"}
        trailing render blob {not json}
    "#;

    let update = extract_tool_update_from_step_payload(6, 8, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["title"], "View README file");
    assert_eq!(update["kind"], "read");
    assert_eq!(update["rawInput"]["AbsolutePath"], "/tmp/project/README.md");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/README.md");
}

#[test]
fn test_extract_tool_update_kind_prefers_tool_name_over_title() {
    let payload = br#"
        view_file
        {"AbsolutePath":"/tmp/project/flow_graph_write_node.go","toolSummary":"View flow_graph_write_node.go"}
    "#;

    let update = extract_tool_update_from_step_payload(7, 8, payload).unwrap();
    assert_eq!(update["title"], "View flow_graph_write_node.go");
    assert_eq!(update["kind"], "read");
}

#[test]
fn test_extract_tool_name_from_embedded_token() {
    assert_eq!(
        extract_tool_name("abc123\tview_file\n{...}"),
        Some("view_file".to_string())
    );
}

#[test]
fn test_extract_tool_update_from_pascal_case_edit_tool() {
    let payload = br#"
        Edit
        {"file_path":"/tmp/project/src/main.rs","old_string":"old","new_string":"new"}
    "#;

    let update = extract_tool_update_from_step_payload(9, 4, payload).unwrap();
    assert_eq!(update["title"], "Edit");
    assert_eq!(update["kind"], "edit");
    assert_eq!(update["rawInput"]["file_path"], "/tmp/project/src/main.rs");
}

#[test]
fn test_extract_tool_update_from_bash_tool() {
    let payload = br#"
        run_command
        {"CommandLine":"cargo test","Cwd":"/tmp/project","toolAction":"Running tests","toolSummary":"Run cargo test"}
    "#;

    let update = extract_tool_update_from_step_payload(10, 21, payload).unwrap();
    assert_eq!(update["title"], "Run cargo test");
    assert_eq!(update["kind"], "execute");
    assert_eq!(update["rawInput"]["CommandLine"], "cargo test");
}

#[test]
fn test_extract_tool_update_from_web_search_step() {
    let payload = br#"
        search_web
        {"query":"FIFA World Cup 2026 dates","toolAction":"Searching World Cup dates","toolSummary":"Search FIFA World Cup 2026 dates"}
    "#;

    let update = extract_tool_update_from_step_payload(3, 33, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["toolCallId"], "agy-3-33");
    assert_eq!(update["title"], "Search FIFA World Cup 2026 dates");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["status"], "in_progress");
    assert_eq!(update["rawInput"]["query"], "FIFA World Cup 2026 dates");
}

#[test]
fn test_extract_tool_update_from_current_agy_generic_tool_step() {
    let payload = br#"
        replace_file_content
        {"TargetFile":"D:\\project\\src\\main.ts","toolAction":"Editing main.ts","toolSummary":"Edit main file"}
    "#;

    let update = extract_tool_update_from_step_payload(57, 132, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["toolCallId"], "agy-57-132");
    assert_eq!(update["title"], "Edit main file");
    assert_eq!(update["kind"], "edit");
    assert_eq!(
        update["rawInput"]["TargetFile"],
        "D:\\project\\src\\main.ts"
    );
}

#[test]
fn test_mcp_receipt_native_policy_denial_is_failed_without_leaking_details() {
    let mut error = Vec::new();
    push_len_field(
        &mut error,
        2,
        b"SLP policy blocked: http://127.0.0.1/mcp/private-capability",
    );
    let payload = make_tool_payload(
        "blocked-consultation",
        "mcp_tool_call",
        r#"{"ToolName":"create_agent","Arguments":{"provider":"codex-peer"},"toolSummary":"Consult Peer"}"#,
        "Consult Peer",
        Some((31, error)),
    );
    let update = extract_tool_update_from_step_payload(1, 140, &payload).unwrap();
    assert_eq!(update["status"], "failed");
    assert_eq!(update["rawOutput"]["isError"], true);
    assert!(!update.to_string().contains("private-capability"));
}

#[test]
fn test_mcp_receipt_success_preserves_child_identity() {
    let receipt = json!({"agentId":"child-1","status":"running","provider":"codex-root"});
    for output in [
        receipt.clone(),
        json!({"content":[{"type":"text","text":receipt.to_string()}]}),
        json!({"jsonrpc":"2.0","id":1,"result":{"structuredContent":receipt}}),
    ] {
        let mut response = Vec::new();
        push_len_field(&mut response, 1, output.to_string().as_bytes());
        let mut mcp = Vec::new();
        push_len_field(&mut mcp, 2, &response);
        let payload = make_tool_payload(
            "create-child",
            "mcp_tool_call",
            r#"{"ToolName":"create_agent","Arguments":{"provider":"codex-root"},"toolSummary":"Assign Lead"}"#,
            "Assign Lead",
            Some((140, mcp)),
        );
        let update = extract_tool_update_from_step_payload(2, 140, &payload).unwrap();
        assert_eq!(update["status"], "completed");
        assert_eq!(
            update["rawOutput"]["receipt"]["agentId"], "child-1",
            "{output}"
        );
        assert_eq!(update["rawOutput"]["receipt"]["provider"], "codex-root");
        let visible = update["content"][0]["content"]["text"].as_str().unwrap();
        assert!(visible.contains("child-1"), "{visible}");
        assert!(visible.contains("codex-root"), "{visible}");
        assert!(visible.contains("running"), "{visible}");
    }
}

#[test]
fn test_mcp_prefixed_receipt_visible_child_facts_are_sanitized() {
    let mut response = Vec::new();
    push_len_field(&mut response, 1, br#"native_metadata=withheld

{"agentId":"child-visible","status":"running","provider":"codex-peer","detail":"private-capability","headers":{"Authorization":"Bearer private-capability"}}"#);
    let mut mcp = Vec::new();
    push_len_field(&mut mcp, 2, &response);
    let payload = make_tool_payload(
        "create-child-visible",
        "call_mcp_tool",
        r#"{"ToolName":"create_agent","toolSummary":"Create Advice Peer"}"#,
        "Create Advice Peer",
        Some((140, mcp)),
    );
    let update = extract_tool_update_from_step_payload(2, 140, &payload).unwrap();
    assert_eq!(update["status"], "completed");
    let visible = update["content"][0]["content"]["text"].as_str().unwrap();
    assert!(visible.contains("child-visible"), "{visible}");
    assert!(visible.contains("running"), "{visible}");
    assert!(!update.to_string().contains("private-capability"));
}

#[test]
fn test_mcp_receipt_structured_error_is_not_success() {
    for output in [
        json!({"isError":true,"content":[{"type":"text","text":"Denied"}]}),
        json!({"jsonrpc":"2.0","id":1,"result":{
            "isError":true,
            "structuredContent":{"agentId":"partial-child","detail":"private-capability"}
        }}),
    ] {
        let mut response = Vec::new();
        push_len_field(&mut response, 1, output.to_string().as_bytes());
        let mut mcp = Vec::new();
        push_len_field(&mut mcp, 2, &response);
        let payload = make_tool_payload(
            "denied-child",
            "mcp_tool_call",
            r#"{"ToolName":"create_agent","toolSummary":"Assign Lead"}"#,
            "Assign Lead",
            Some((140, mcp)),
        );
        let update = extract_tool_update_from_step_payload(3, 140, &payload).unwrap();
        assert_eq!(update["status"], "failed", "{output}");
        assert!(!update.to_string().contains("private-capability"));
    }
}

#[test]
fn test_mcp_receipt_turn_completion_does_not_invent_tool_success() {
    let state = Arc::new(Mutex::new(StreamingState::default()));
    {
        let mut guard = state.lock().unwrap();
        guard
            .tool_call_statuses
            .insert("pending".into(), "in_progress".into());
        guard
            .tool_call_statuses
            .insert("known-success".into(), "completed".into());
        guard
            .tool_call_statuses
            .insert("known-denial".into(), "failed".into());
    }
    let lines = finalize_pending_tool_calls("session-1", &state, "completed");
    assert_eq!(lines.len(), 1);
    let notification: Value = serde_json::from_str(&lines[0]).unwrap();
    let update = &notification["params"]["update"];
    assert_eq!(update["toolCallId"], "pending");
    assert_eq!(update["status"], "failed");
    assert_eq!(update["rawOutput"]["resultType"], "unverified");
    assert_eq!(
        state.lock().unwrap().tool_call_statuses["known-success"],
        "completed"
    );
    assert_eq!(
        state.lock().unwrap().tool_call_statuses["known-denial"],
        "failed"
    );
}

#[test]
fn test_mcp_receipt_streaming_preserves_success_when_parent_turn_fails() {
    let root = std::env::temp_dir().join(format!("agy-acp-mcp-receipt-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let conn = Connection::open(root.join("receipt.db")).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, step_payload BLOB)",
    )
    .unwrap();
    let input = r#"{"ToolName":"create_agent","toolSummary":"Assign Lead"}"#;
    let pending = make_tool_payload("child-call", "mcp_tool_call", input, "Assign Lead", None);
    conn.execute("INSERT INTO steps VALUES (1,140,?1)", [pending])
        .unwrap();
    let state = Arc::new(Mutex::new(StreamingState {
        conversation_id: Some("receipt".into()),
        base_step_idx: -1,
        suppress_agent_text: true,
        ..StreamingState::default()
    }));
    let first = poll_streaming_delta(&root, None, "session-1", &state);
    assert_eq!(first.len(), 1);
    assert_eq!(
        state.lock().unwrap().tool_call_statuses["child-call"],
        "in_progress"
    );
    let mut response = Vec::new();
    push_len_field(
        &mut response,
        1,
        br#"{"agentId":"child-2","status":"running"}"#,
    );
    let mut mcp = Vec::new();
    push_len_field(&mut mcp, 2, &response);
    let completed = make_tool_payload(
        "child-call",
        "mcp_tool_call",
        input,
        "Assign Lead",
        Some((140, mcp)),
    );
    conn.execute("INSERT INTO steps VALUES (2,140,?1)", [completed])
        .unwrap();
    drop(conn);
    let second = poll_streaming_delta(&root, None, "session-1", &state);
    assert_eq!(second.len(), 1);
    let update: Value = serde_json::from_str(&second[0]).unwrap();
    assert_eq!(
        update["params"]["update"]["sessionUpdate"],
        "tool_call_update"
    );
    assert_eq!(update["params"]["update"]["status"], "completed");
    assert!(finalize_pending_tool_calls("session-1", &state, "failed").is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_streaming_discovers_structured_tool_payload_when_step_type_changes() {
    let root = std::env::temp_dir().join(format!("agy-acp-tool-type-drift-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let conn = Connection::open(conv_dir.join("conv-type-drift.db")).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, ?2, ?3)",
        rusqlite::params![
            1i64,
            144i64,
            br#"replace_file_content
            {"TargetFile":"D:\\project\\src\\main.ts","toolAction":"Editing main.ts","toolSummary":"Edit main file"}"#
                .as_slice()
        ],
    )
    .unwrap();
    drop(conn);

    let state = Arc::new(Mutex::new(StreamingState {
        conversation_id: Some("conv-type-drift".to_string()),
        base_step_idx: -1,
        suppress_agent_text: true,
        ..StreamingState::default()
    }));
    let notifications = poll_streaming_delta(&conv_dir, None, "session-1", &state);

    assert_eq!(
        notifications.len(),
        1,
        "a structurally valid tool payload must stream even when Agy changes its numeric step type"
    );
    let notification: Value = serde_json::from_str(&notifications[0]).unwrap();
    assert_eq!(
        notification["params"]["update"]["sessionUpdate"],
        "tool_call"
    );
    assert_eq!(notification["params"]["update"]["title"], "Edit main file");

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_streaming_rebinds_when_native_replaces_a_missing_conversation() {
    let root = std::env::temp_dir().join(format!("agy-acp-stream-rebind-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();
    let replacement_id = Uuid::new_v4().to_string();
    let run_log = root.join("run.log");
    fs::write(
        &run_log,
        format!(
            "Conversation {} not found, ignoring --conversation flag\nCreated conversation {}\n",
            Uuid::new_v4(),
            replacement_id
        ),
    )
    .unwrap();

    let conn = Connection::open(conv_dir.join(format!("{replacement_id}.db"))).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, ?2, ?3)",
        rusqlite::params![
            1i64,
            144i64,
            br#"run_command
            {"CommandLine":"mvn test","toolAction":"Running tests","toolSummary":"Run backend tests"}"#
                .as_slice()
        ],
    )
    .unwrap();
    drop(conn);

    let stale_id = Uuid::new_v4().to_string();
    let state = Arc::new(Mutex::new(StreamingState {
        conversation_id: Some(stale_id),
        base_step_idx: 900,
        last_step_idx: 900,
        suppress_agent_text: true,
        ..StreamingState::default()
    }));
    let notifications = poll_streaming_delta(&conv_dir, Some(&run_log), "session-rebind", &state);

    assert_eq!(
        state.lock().unwrap().conversation_id.as_deref(),
        Some(replacement_id.as_str()),
        "the poller must follow the conversation Agy actually created"
    );
    assert_eq!(
        notifications.len(),
        1,
        "replacement conversation updates must stream"
    );
    let notification: Value = serde_json::from_str(&notifications[0]).unwrap();
    assert_eq!(
        notification["params"]["update"]["sessionUpdate"],
        "tool_call"
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_extract_tool_update_maps_reasoning_to_think_content() {
    let payload = br#"
        thinking
        {"thought":"Need to inspect the protocol before changing serialization.","toolSummary":"Reasoning"}
    "#;

    let update = extract_tool_update_from_step_payload(21, 17, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "agent_thought_chunk");
    assert_eq!(update["content"]["type"], "text");
    assert_eq!(
        update["content"]["text"],
        "Need to inspect the protocol before changing serialization."
    );
}

#[test]
fn test_extract_tool_update_from_structured_grep_payload() {
    let mut grep = Vec::new();
    push_len_field(&mut grep, 1, b"StepPayload");
    push_len_field(&mut grep, 2, b"src/*.rs");
    push_len_field(&mut grep, 3, b"src/protobuf.rs:1:message StepPayload");
    push_len_field(&mut grep, 10, b"rg StepPayload src");
    push_len_field(&mut grep, 11, b"file:///tmp/project");
    let payload = make_tool_payload(
        "0t0p5kn3",
        "grep_search",
        r#"{"SearchPath":"/tmp/project/src","toolAction":"Searching protobuf schema"}"#,
        "Proto search",
        Some((13, grep)),
    );

    let update = extract_tool_update_from_step_payload(22, 7, &payload).unwrap();
    assert_eq!(update["toolCallId"], "0t0p5kn3");
    assert_eq!(update["title"], "Proto search");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["rawInput"]["SearchPath"], "/tmp/project/src");
    assert_eq!(update["rawOutput"]["query"], "StepPayload");
    assert_eq!(
        update["rawOutput"]["textOutput"],
        "src/protobuf.rs:1:message StepPayload"
    );
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src");
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nsrc/protobuf.rs:1:message StepPayload\n```"
    );
}

#[test]
fn test_extract_tool_update_formats_structured_grep_hits_without_text_output() {
    let mut hit = Vec::new();
    push_len_field(&mut hit, 1, b"src/protobuf.rs");
    push_varint_field(&mut hit, 2, 42);
    push_len_field(
        &mut hit,
        3,
        b"fn parse_tool_result(blob: &[u8]) -> Option<Value> {",
    );

    let mut grep = Vec::new();
    push_len_field(&mut grep, 1, b"parse_tool_result");
    push_len_field(&mut grep, 4, &hit);
    let payload = make_tool_payload(
        "grep-hit-call",
        "grep_search",
        r#"{"SearchPath":"/tmp/project/src","toolAction":"Searching parser"}"#,
        "Parser search",
        Some((13, grep)),
    );

    let update = extract_tool_update_from_step_payload(26, 7, &payload).unwrap();
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nfield1: src/protobuf.rs | field2: 42 | field3: fn parse_tool_result(blob: &[u8]) -> Option<Value> {\n```"
    );
}

#[test]
fn test_extract_tool_update_from_structured_view_payload() {
    let mut view = Vec::new();
    push_len_field(&mut view, 1, b"file:///tmp/project/src/protobuf.rs");
    push_varint_field(&mut view, 2, 10);
    push_varint_field(&mut view, 3, 12);
    push_len_field(&mut view, 4, b"pub fn read_varint() {}\n```");
    push_varint_field(&mut view, 11, 13);
    push_varint_field(&mut view, 12, 200);
    let payload = make_tool_payload(
        "view-call",
        "view_file",
        "{}",
        "Viewing file",
        Some((14, view)),
    );

    let update = extract_tool_update_from_step_payload(23, 8, &payload).unwrap();
    assert_eq!(update["title"], "Viewing file");
    assert_eq!(update["kind"], "read");
    assert_eq!(
        update["rawOutput"]["fileUri"],
        "file:///tmp/project/src/protobuf.rs"
    );
    assert_eq!(update["rawOutput"]["startLine"], 10);
    assert_eq!(
        update["locations"][0]["path"],
        "file:///tmp/project/src/protobuf.rs"
    );
    assert_eq!(update["locations"][0]["line"], 10);
    assert_eq!(
        update["content"][0]["content"]["text"],
        "````\npub fn read_varint() {}\n```\n````"
    );
}

#[test]
fn test_extract_tool_update_from_structured_list_payload() {
    let mut entry = Vec::new();
    push_len_field(&mut entry, 1, b"src");
    push_varint_field(&mut entry, 2, 1);
    push_varint_field(&mut entry, 4, 0);

    let mut list = Vec::new();
    push_len_field(&mut list, 1, b"file:///tmp/project");
    push_len_field(&mut list, 3, &entry);
    let payload = make_tool_payload(
        "list-call",
        "list_dir",
        "{}",
        "Listing directory",
        Some((15, list)),
    );

    let update = extract_tool_update_from_step_payload(24, 9, &payload).unwrap();
    assert_eq!(update["title"], "Listing directory");
    assert_eq!(update["kind"], "read");
    assert_eq!(update["rawOutput"]["dirUri"], "file:///tmp/project");
    assert_eq!(update["rawOutput"]["entries"][0]["name"], "src");
    assert_eq!(update["rawOutput"]["entries"][0]["isDirectory"], true);
    assert_eq!(update["content"][0]["content"]["text"], "```\nsrc/\n```");
}

#[test]
fn test_extract_tool_update_formats_empty_structured_list_payload() {
    let mut list = Vec::new();
    push_len_field(&mut list, 1, b"file:///tmp/project");
    let payload = make_tool_payload(
        "empty-list-call",
        "list_dir",
        "{}",
        "Listing directory",
        Some((15, list)),
    );

    let update = extract_tool_update_from_step_payload(27, 9, &payload).unwrap();
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\n(empty directory)\n```"
    );
}

#[test]
fn test_extract_tool_update_from_structured_write_payload() {
    let mut write = Vec::new();
    push_len_field(&mut write, 26, b"Wrote 42 bytes");
    let payload = make_tool_payload(
        "write-call",
        "write_to_file",
        r#"{"AbsolutePath":"/tmp/project/src/main.rs"}"#,
        "Writing file",
        Some((10, write)),
    );

    let update = extract_tool_update_from_step_payload(25, 5, &payload).unwrap();
    assert_eq!(update["title"], "Writing file");
    assert_eq!(update["kind"], "edit");
    assert_eq!(update["rawOutput"]["summary"], "Wrote 42 bytes");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src/main.rs");
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nWrote 42 bytes\n```"
    );
}

#[test]
fn test_read_varint() {
    assert_eq!(read_varint(&[0x05]), Some((5, 1)));
    assert_eq!(read_varint(&[0xAC, 0x02]), Some((300, 2)));
    assert_eq!(read_varint(&[]), None);
}

#[test]
fn test_initialize_advertises_load_session_support() {
    let adapter = adapter_for_test();
    let response = adapter.handle_initialize(json!(1));
    assert_eq!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("agentCapabilities"))
            .and_then(|c| c.get("loadSession"))
            .and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_initialize_advertises_resume_capability() {
    let adapter = adapter_for_test();
    let response = adapter.handle_initialize(json!(1));
    assert!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("agentCapabilities"))
            .and_then(|c| c.get("sessionCapabilities"))
            .and_then(|sc| sc.get("resume"))
            .is_some(),
        "sessionCapabilities.resume should be present"
    );
}

#[test]
#[ignore]
fn test_session_load_restores_persisted_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-load-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.persist_session("sess-1", Some("conv-abc"), 5, None);

    let output = adapter.handle_session_load(json!(7), &json!({"sessionId": "sess-1"}));
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(
        adapter
            .sessions
            .get("sess-1")
            .and_then(|s| s.conversation_id.as_deref()),
        Some("conv-abc")
    );
    assert_eq!(
        adapter.sessions.get("sess-1").map(|s| s.last_step_idx),
        Some(5)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_load_rejects_unknown_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-missing-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };

    let output = adapter.handle_session_load(json!(9), &json!({"sessionId": "missing"}));
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response["result"].is_null());
    assert_eq!(
        response["error"]["message"].as_str(),
        Some("unknown sessionId: missing")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_load_replays_conversation_history() {
    let root = std::env::temp_dir().join(format!("agy-acp-load-replay-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let db_path = conv_dir.join("conv-replay.db");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
        rusqlite::params![1i64, make_user_payload("hello")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![
            2i64,
            make_assistant_payload("I will inspect the workspace.")
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 8, ?2)",
        rusqlite::params![
            3i64,
            br#"view_file
            {"AbsolutePath":"/tmp/project/README.md","toolAction":"Reading README.md","toolSummary":"View README file"}
            trailing render blob {not json}"#
                .as_slice()
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 5, ?2)",
        rusqlite::params![
            4i64,
            br#"replace_file_content
            {"AbsolutePath":"/tmp/project/README.md","toolAction":"Editing README.md","toolSummary":"Edit README file"}"#
                .as_slice()
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 21, ?2)",
        rusqlite::params![
            5i64,
            br#"run_command
            {"CommandLine":"cargo test","Cwd":"/tmp/project","toolAction":"Running tests","toolSummary":"Run cargo test"}"#
                .as_slice()
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 17, ?2)",
        rusqlite::params![6i64, br#"thinking
            {"thought":"I should verify the test output before summarizing.","toolSummary":"Analyzing test output"}"#
            .as_slice()],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![7i64, make_assistant_payload("hello from agent")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
        rusqlite::params![8i64, make_user_payload("how are you?")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![9i64, make_assistant_payload("second response")],
    )
    .unwrap();
    drop(conn);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: conv_dir,
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.persist_session("sess-replay", Some("conv-replay"), 9, None);

    let output = adapter.handle_session_load(json!(1), &json!({"sessionId": "sess-replay"}));

    assert!(
        output.len() >= 2,
        "expected replay notification + response, got {}",
        output.len()
    );

    let updates: Vec<Value> = output[..output.len() - 1]
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(updates.iter().any(|notification| {
        notification["method"] == "session/update"
            && notification["params"]["update"]["sessionUpdate"] == "tool_call"
            && notification["params"]["update"]["title"] == "View README file"
            && notification["params"]["update"]["kind"] == "read"
    }));
    assert!(updates.iter().any(|notification| {
        notification["params"]["update"]["title"] == "Edit README file"
            && notification["params"]["update"]["kind"] == "edit"
    }));
    assert!(updates.iter().any(|notification| {
        notification["params"]["update"]["title"] == "Run cargo test"
            && notification["params"]["update"]["kind"] == "execute"
    }));
    let replay_kinds: Vec<_> = updates
        .iter()
        .map(|notification| {
            notification["params"]["update"]["sessionUpdate"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(
        replay_kinds,
        vec![
            "user_message_chunk",
            "agent_message_chunk",
            "tool_call",
            "tool_call",
            "tool_call",
            "agent_thought_chunk",
            "agent_message_chunk",
            "user_message_chunk",
            "agent_message_chunk"
        ]
    );
    let message_updates: Vec<_> = updates
        .iter()
        .filter(|notification| {
            matches!(
                notification["params"]["update"]["sessionUpdate"].as_str(),
                Some("user_message_chunk")
                    | Some("agent_message_chunk")
                    | Some("agent_thought_chunk")
            )
        })
        .collect();
    let update_kinds: Vec<_> = message_updates
        .iter()
        .map(|notification| {
            notification["params"]["update"]["sessionUpdate"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(
        update_kinds,
        vec![
            "user_message_chunk",
            "agent_message_chunk",
            "agent_thought_chunk",
            "agent_message_chunk",
            "user_message_chunk",
            "agent_message_chunk"
        ]
    );
    let message_texts: Vec<_> = message_updates
        .iter()
        .map(|notification| {
            notification["params"]["update"]["content"]["text"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(
        message_texts,
        vec![
            "hello",
            "I will inspect the workspace.",
            "I should verify the test output before summarizing.",
            "hello from agent",
            "how are you?",
            "second response"
        ]
    );
    assert!(
        message_texts[1].contains("I will inspect"),
        "load replay should preserve narration shown in the live session"
    );

    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(
        response["result"]["sessionId"].as_str(),
        Some("sess-replay")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_resume_restores_persisted_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.persist_session("sess-r1", Some("conv-xyz"), 3, None);

    let response = adapter.handle_session_resume(json!(10), &json!({"sessionId": "sess-r1"}));
    assert!(response.error.is_none());
    assert_eq!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("sessionId"))
            .and_then(|s| s.as_str()),
        Some("sess-r1")
    );
    assert_eq!(
        adapter
            .sessions
            .get("sess-r1")
            .and_then(|s| s.conversation_id.as_deref()),
        Some("conv-xyz")
    );
    assert_eq!(
        adapter.sessions.get("sess-r1").map(|s| s.last_step_idx),
        Some(3)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_resume_rejects_unknown_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-miss-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };

    let response = adapter.handle_session_resume(json!(11), &json!({"sessionId": "nope"}));
    assert!(response.result.is_none());
    assert_eq!(
        response
            .error
            .as_ref()
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str()),
        Some("unknown sessionId: nope")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_session_resume_rejects_empty_session_id() {
    let mut adapter = adapter_for_test();
    let response = adapter.handle_session_resume(json!(12), &json!({}));
    assert!(response.result.is_none());
    assert_eq!(
        response
            .error
            .as_ref()
            .and_then(|e| e.get("code"))
            .and_then(|c| c.as_i64()),
        Some(-32602)
    );
}

#[test]
fn test_session_resume_accepts_in_memory_session() {
    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: "/tmp".to_string(),
        conversations_dir: PathBuf::from("/tmp/conversations"),
        state_file: PathBuf::from("/tmp/nonexistent-agy-acp-sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.sessions.insert(
        "sess-memory".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
            mode_id: crate::types::default_mode_id(),
            caller_context: json!({}),
            mcp_servers: json!([]),
        },
    );

    let response = adapter.handle_session_resume(json!(12), &json!({"sessionId": "sess-memory"}));

    assert!(response.error.is_none());
    assert_eq!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("sessionId"))
            .and_then(|s| s.as_str()),
        Some("sess-memory")
    );
}

#[test]
fn test_session_load_accepts_in_memory_session_without_replay() {
    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: "/tmp".to_string(),
        conversations_dir: PathBuf::from("/tmp/conversations"),
        state_file: PathBuf::from("/tmp/nonexistent-agy-acp-sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.sessions.insert(
        "sess-memory-load".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
            mode_id: crate::types::default_mode_id(),
            caller_context: json!({}),
            mcp_servers: json!([]),
        },
    );

    let output = adapter.handle_session_load(json!(13), &json!({"sessionId": "sess-memory-load"}));

    assert_eq!(output.len(), 1);
    let response: Value = serde_json::from_str(&output[0]).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(response["result"]["sessionId"], "sess-memory-load");
}

#[test]
#[ignore]
fn test_session_resume_does_not_replay_history() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-noreplay-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.persist_session("sess-nr", Some("conv-nr"), 10, None);

    let response = adapter.handle_session_resume(json!(13), &json!({"sessionId": "sess-nr"}));
    assert!(response.error.is_none());
    assert_eq!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("sessionId"))
            .and_then(|s| s.as_str()),
        Some("sess-nr")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_finds_conversation_from_invocation_specific_log() {
    let root = std::env::temp_dir().join(format!("agy-acp-log-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    fs::write(
        &log_path,
        b"I0722 printmode.go] conversationID=\"\"\nI0722 server.go] Created conversation 155c3eac-60e1-43c6-8f7c-eee98a34c2b8\n",
    )
    .unwrap();

    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&log_path),
        Some("155c3eac-60e1-43c6-8f7c-eee98a34c2b8".to_string())
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_rejects_invalid_conversation_id_in_invocation_log() {
    let root = std::env::temp_dir().join(format!("agy-acp-log-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    fs::write(
        &log_path,
        b"I0722 server.go] Created conversation not-a-uuid\n",
    )
    .unwrap();

    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&log_path),
        None
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_parallel_invocation_logs_cannot_cross_bind() {
    let root = std::env::temp_dir().join(format!("agy-acp-parallel-logs-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let first_log = root.join("first.log");
    let second_log = root.join("second.log");
    fs::write(
        &first_log,
        b"I0722 server.go] Created conversation 155c3eac-60e1-43c6-8f7c-eee98a34c2b8\n",
    )
    .unwrap();
    fs::write(
        &second_log,
        b"I0722 server.go] Created conversation 3f33f7fe-787a-489c-9fa9-477e6b4bf19b\n",
    )
    .unwrap();

    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&first_log).as_deref(),
        Some("155c3eac-60e1-43c6-8f7c-eee98a34c2b8")
    );
    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&second_log).as_deref(),
        Some("3f33f7fe-787a-489c-9fa9-477e6b4bf19b")
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_detects_untrustworthy_print_timeout() {
    let root = std::env::temp_dir().join(format!("agy-acp-timeout-log-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    fs::write(
        &log_path,
        b"I0722 printmode.go] Print mode: timed out after 1496 polls (printed=162)\n",
    )
    .unwrap();

    assert!(crate::db::agy_run_timed_out(&log_path));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_captured_native_stream_lag_replay_is_classified_without_replay() {
    let root = std::env::temp_dir().join(format!("agy-acp-stream-lag-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    let captured_signature = "ERROR: logging before google.Init: W0904 11:15:58.504824     302 pubsub.go:78] pubsub: Publish killing slow subscriber \"11111111-1111-4111-8111-111111111111\": subscriber fell behind updates, stalled for 5s\nERROR: logging before google.Init: E0904 11:15:58.869216     154 server.go:1215] StreamAgentStateUpdates error for 22222222-2222-4222-8222-222222222222: subscriber fell behind updates, stalled for 5s\nERROR: logging before google.Init: E0904 11:15:58.870223     154 conversation_manager.go:978] Conversation update stream for 22222222-2222-4222-8222-222222222222 failed: subscriber fell behind updates, stalled for 5s\nERROR: logging before google.Init: I0904 11:15:58.871122       1 printmode.go:545] Print mode: conversation update stream failed: subscriber fell behind updates, stalled for 5s\n";
    fs::write(&log_path, captured_signature).unwrap();

    let output = Adapter::prompt_failure_for_run_log(
        "sess-stream-lag",
        json!(42),
        &log_path,
        "agy exited with status: exit code: 1 and reported an error",
    );
    let serialized = output.join("\n");
    let update: Value = serde_json::from_str(&output[0]).unwrap();
    let text = update["params"]["update"]["content"]["text"]
        .as_str()
        .unwrap();
    assert!(
        text.contains("AGY_NATIVE_STREAM_LAG"),
        "captured stream-lag replay was not classified: {serialized}"
    );
    assert!(text.contains("code: -32000"));
    assert!(text.contains("\"kind\":\"AGY_NATIVE_STREAM_LAG\""));
    assert!(text.contains("\"code\":\"AGY_NATIVE_STREAM_LAG\""));
    assert!(text.contains("\"acpCode\":-32000"));
    let escaped_run_log = log_path.display().to_string().replace('\\', "\\\\");
    assert!(text.contains(&escaped_run_log));
    assert!(text.contains("side_effects_possible_no_replay"));
    assert!(text.contains("do not replay"));
    assert!(text.contains("fresh Agy conversation/seat"));
    assert!(text.contains("bounded context handoff"));
    assert!(text.contains("preserve the old conversation and run log"));
    assert!(!text.contains("pubsub.go"));
    assert!(!text.contains("subscriber fell behind updates"));

    let missing_stream_marker = root.join("missing-stream-marker.log");
    fs::write(
        &missing_stream_marker,
        captured_signature.replace("StreamAgentStateUpdates error", "StreamAgentState status"),
    )
    .unwrap();
    let generic_output = Adapter::prompt_failure_for_run_log(
        "sess-stream-lag",
        json!(43),
        &missing_stream_marker,
        "agy exited with status: exit code: 1 and reported an error",
    );
    let generic_text = generic_output.join("\n");
    assert!(!generic_text.contains("AGY_NATIVE_STREAM_LAG"));
    assert!(generic_text.contains("agy exited with status: exit code: 1"));

    let upstream_six_second_signature = root.join("upstream-six-second-signature.log");
    fs::write(
        &upstream_six_second_signature,
        captured_signature.replace("stalled for 5s", "stalled for 6s"),
    )
    .unwrap();
    let upstream_output = Adapter::prompt_failure_for_run_log(
        "sess-stream-lag",
        json!(44),
        &upstream_six_second_signature,
        "agy exited with status: exit code: 1 and reported an error",
    );
    assert!(upstream_output.join("\n").contains("AGY_NATIVE_STREAM_LAG"));

    let different_cause = root.join("different-cause.log");
    fs::write(
        &different_cause,
        captured_signature.replace("subscriber fell behind updates", "subscriber disconnected"),
    )
    .unwrap();
    let different_cause_output = Adapter::prompt_failure_for_run_log(
        "sess-stream-lag",
        json!(45),
        &different_cause,
        "agy exited with status: exit code: 1 and reported an error",
    );
    assert!(!different_cause_output
        .join("\n")
        .contains("AGY_NATIVE_STREAM_LAG"));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_quota_exhaustion_surfaces_the_provider_reset_window() {
    let root = std::env::temp_dir().join(format!("agy-acp-quota-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    fs::write(
        &log_path,
        "E1005 errorreport.go:224] generating and executing: RESOURCE_EXHAUSTED (code 429): Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 45m13s.\n\
         E1005 session.go:271] Print mode: run ended with error and no response: Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 45m13s.\n",
    )
    .unwrap();

    let output = Adapter::prompt_failure_for_run_log(
        "sess-quota",
        json!(46),
        &log_path,
        "agy exited with status: exit code: 3 and reported an error",
    );
    let text = output.join("\n");
    assert!(text.contains("AGY_MODEL_QUOTA_EXHAUSTED"));
    assert!(text.contains("45m13s"));
    assert!(!text.contains("agy exited with status: exit code: 3"));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_captured_eligibility_network_eof_is_classified_as_safe_to_retry() {
    let root = std::env::temp_dir().join(format!("agy-acp-eligibility-eof-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    fs::write(
        &log_path,
        b"I0910 17:33:09.492797 auth.go:148] ChainedAuth: authenticated via keyring (effective: keyring)\n\
I0910 17:33:09.493299 server_oauth.go:197] OAuth: authenticated successfully\n\
E0910 17:33:32.734937 error_parsing.go:65] Error: Get \"https://www.googleapis.com/oauth2/v2/userinfo\": EOF\n\
E0910 17:33:32.734937 server_oauth.go:85] Validation failed: Get \"https://www.googleapis.com/oauth2/v2/userinfo\": EOF\n\
E0910 17:33:32.747678 printmode.go:253] Print mode: eligibility check failed: Eligibility check failed: Get \"https://www.googleapis.com/oauth2/v2/userinfo\": EOF\n",
    )
    .unwrap();

    let kind = crate::db::classify_agy_run_failure(&log_path)
        .expect("captured eligibility EOF must be classified");
    assert_eq!(kind.stable_code(), "AGY_ELIGIBILITY_NETWORK_TRANSIENT");
    assert!(kind.safe_to_retry_before_output());
    assert!(Adapter::should_retry_run_failure(Some(kind), 0, false));
    assert!(Adapter::should_retry_run_failure(Some(kind), 1, false));
    assert!(!Adapter::should_retry_run_failure(Some(kind), 2, false));
    assert!(!Adapter::should_retry_run_failure(Some(kind), 0, true));
    assert!(!Adapter::should_retry_run_failure(
        Some(crate::db::AgyRunFailureKind::NativeStreamLag),
        0,
        false,
    ));

    let v135_log_path = root.join("run-1.3.5-503.log");
    fs::write(
        &v135_log_path,
        b"W1014 16:08:43.578979 1 printmode.go:518] Print mode: eligibility verdict did not settle within 1m0s; proceeding\n\
W1014 16:08:44.617769 583 cache.go:135] Cache(loadCodeAssistResponse): Singleflight refresh failed: UNAVAILABLE (code 503): The service is currently unavailable.\n\
W1014 16:08:44.622795 1 conversation_manager.go:762] Not sending user message: Eligibility check failed: failed to get load code assist response: UNAVAILABLE (code 503): The service is currently unavailable.\n\
E1014 16:08:44.622795 1 session.go:207] Print mode: SendUserMessage failed: send failed; already reported to the user: Eligibility check failed: failed to get load code assist response: UNAVAILABLE (code 503): The service is currently unavailable.\n",
    )
    .unwrap();
    let v135_kind = crate::db::classify_agy_run_failure(&v135_log_path)
        .expect("Agy 1.3.5 eligibility 503 failure must be classified");
    assert_eq!(v135_kind.stable_code(), "AGY_ELIGIBILITY_NETWORK_TRANSIENT");
    assert!(Adapter::should_retry_run_failure(Some(v135_kind), 0, false));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_detects_headless_permission_denial_markers() {
    let root = std::env::temp_dir().join(format!("agy-acp-permission-log-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let soft_deny_log = root.join("soft-deny.log");
    let failed_check_log = root.join("failed-check.log");
    fs::write(
        &soft_deny_log,
        b"I0722 printmode.go] Print mode: soft-denying tool confirmation for Edit\n",
    )
    .unwrap();
    fs::write(
        &failed_check_log,
        b"E0722 permissions.go] permission check failed: write_file requires confirmation\n",
    )
    .unwrap();

    assert!(crate::db::agy_run_permission_denied(&soft_deny_log));
    assert!(crate::db::agy_run_permission_denied(&failed_check_log));
    assert!(!crate::db::agy_run_timed_out(&soft_deny_log));

    let diagnostic = Adapter::no_assistant_response_message(&soft_deny_log);
    assert!(diagnostic.contains("permission denied"));
    assert!(diagnostic.contains("accept-edits"));
    assert!(diagnostic.contains("plan"));
    assert!(diagnostic.contains("Direct Paseo sessions honor the selected mode"));
    assert!(diagnostic.contains("delegated worker assignments still require Lead approval"));
    assert_ne!(diagnostic, "agy completed without an assistant response");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_prompt_failure_is_model_visible_instead_of_json_rpc_error() {
    let output = Adapter::prompt_failure_lines(
        "sess-permission",
        json!(42),
        -32000,
        "agy permission denied",
        Some(json!({"operation": "write_file"})),
    );

    assert_eq!(output.len(), 2);
    let update: Value = serde_json::from_str(&output[0]).unwrap();
    assert_eq!(update["method"], "session/update");
    let text = update["params"]["update"]["content"]["text"]
        .as_str()
        .unwrap();
    assert!(text.contains("[System Error] agy permission denied"));
    assert!(text.contains("code: -32000"));
    assert!(text.contains("data: {\"operation\":\"write_file\"}"));

    let response: Value = serde_json::from_str(&output[1]).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

#[test]
fn test_prompt_failure_redacts_mcp_credentials_from_output() {
    let secret = "stage2-diagnostic-secret";
    let output = Adapter::prompt_failure_lines(
        "sess-redaction",
        json!(42),
        -32001,
        "required MCP attachment failed",
        Some(json!({
            "mcpServers": [{"env": {"API_TOKEN": secret}}],
            "operation": "attach"
        })),
    );

    let serialized = output.join("\n");
    assert!(!serialized.contains(secret));
    assert!(serialized.contains("[REDACTED]"));
    let update: Value = serde_json::from_str(&output[0]).unwrap();
    let text = update["params"]["update"]["content"]["text"]
        .as_str()
        .unwrap();
    assert!(text.contains("\"operation\":\"attach\""));
}

#[test]
fn test_prompt_session_merge_is_bounded_to_completed_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-prompt-merge-{}", Uuid::new_v4()));
    let mut shared = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    shared.sessions.insert(
        "session-a".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: Some("Gemini 3.7 Flash (High)".to_string()),
            mode_id: "plan".to_string(),
            caller_context: json!({}),
            mcp_servers: json!([]),
        },
    );
    shared.sessions.insert(
        "session-b".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: Some("Gemini 3.7 Flash (High)".to_string()),
            mode_id: "plan".to_string(),
            caller_context: json!({}),
            mcp_servers: json!([]),
        },
    );

    let mut completed = shared.clone();
    completed
        .sessions
        .get_mut("session-a")
        .unwrap()
        .conversation_id = Some("conversation-a".to_string());
    completed
        .sessions
        .get_mut("session-a")
        .unwrap()
        .last_step_idx = 7;
    completed.sessions.get_mut("session-b").unwrap().mode_id = "accept-edits".to_string();

    shared.merge_prompt_session_from("session-a", &completed);

    assert_eq!(
        shared.sessions["session-a"].conversation_id.as_deref(),
        Some("conversation-a")
    );
    assert_eq!(shared.sessions["session-a"].last_step_idx, 7);
    assert_eq!(shared.sessions["session-b"].mode_id, "plan");
}

#[test]
fn test_native_model_catalog_precedes_stale_materialized_fallback() {
    let stale =
        r#"["gemini-3.7-flash-high\tGemini 3.7 Flash (High)","Claude Sonnet 4.6 (Thinking)"]"#;
    let native = concat!(
        "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n",
        "gemini-3.7-flash-high\tGemini 3.7 Flash (High)\n"
    );

    let models = Adapter::resolve_available_models(Some(native), Some(stale));
    assert_eq!(
        models,
        vec![
            "Gemini 3.8 Flash (Low)".to_string(),
            "Gemini 3.7 Flash (High)".to_string(),
        ]
    );

    let fallback = Adapter::resolve_available_models(None, Some(stale));
    assert_eq!(fallback[0], "Gemini 3.7 Flash (High)");

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: String::new(),
        conversations_dir: PathBuf::new(),
        state_file: PathBuf::new(),
        available_models: models,
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    let fresh = adapter.session_models_json(None);
    assert_eq!(fresh["currentModelId"], "Gemini 3.8 Flash (Low)");
    let durable = adapter.session_models_json(Some("Gemini 3.6 Flash (High)"));
    assert_eq!(durable["currentModelId"], "Gemini 3.6 Flash (High)");
    assert_eq!(
        durable["availableModels"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["modelId"],
        "Gemini 3.6 Flash (High)"
    );
    assert!(Adapter::parse_materialized_models("[]").is_none());
    assert!(Adapter::parse_materialized_models("not-json").is_none());
}

#[test]
fn test_model_cache_never_reuses_another_accounts_catalog() {
    let mut cache = crate::adapter::ModelCache::default();
    assert_eq!(
        cache.update(Some(1), vec!["Account A".into()]),
        vec!["Account A"]
    );
    assert!(cache.update(Some(2), vec![]).is_empty());
    assert_eq!(
        cache.update(Some(2), vec!["Account B".into()]),
        vec!["Account B"]
    );
    assert_eq!(cache.update(Some(1), vec![]), vec!["Account A"]);
    assert_eq!(cache.update(Some(2), vec![]), vec!["Account B"]);
}

#[test]
fn test_generated_agy_argv_contains_exact_selected_mode() {
    let root = std::env::temp_dir().join(format!("agy-acp-mode-argv-{}", Uuid::new_v4()));
    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec!["Gemini 3.7 Flash (High)".to_string()],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.sessions.insert(
        "mode-session".to_string(),
        crate::types::Session {
            conversation_id: Some("conversation-1".to_string()),
            last_step_idx: 3,
            model_id: Some("Gemini 3.7 Flash (High)".to_string()),
            mode_id: "plan".to_string(),
            caller_context: json!({}),
            mcp_servers: json!([]),
        },
    );

    let args = adapter.build_agy_args("mode-session", &root.join("run.log"), None);
    let mode_index = args.iter().position(|arg| arg == "--mode").unwrap();
    assert_eq!(args[mode_index + 1], "plan");
    assert_eq!(args.iter().filter(|arg| *arg == "--mode").count(), 1);
    assert!(!args
        .iter()
        .any(|arg| arg == "--dangerously-skip-permissions"));
    assert!(args
        .windows(2)
        .any(|pair| pair == ["--input-format", "stream-json"]));
    assert!(args
        .windows(2)
        .any(|pair| pair == ["--output-format", "stream-json"]));
    assert!(!args.iter().any(|arg| arg == "-p" || arg == "--print"));
    assert!(!args.iter().any(|arg| arg == "Inspect without editing"));
}

#[test]
fn test_large_prompt_is_serialized_to_stream_input_instead_of_argv() {
    let prompt = format!("{}\nTiếng Việt cuối dòng", "x".repeat(70_000));
    let line = Adapter::build_stream_input_line(&prompt).unwrap();
    let decoded: Value = serde_json::from_str(&line).unwrap();

    assert_eq!(decoded["event"], "user");
    assert!(decoded["message"]["content"]
        .as_str()
        .unwrap()
        .ends_with(&prompt));
    assert!(line.len() > 70_000);
}

#[test]
fn test_stream_prompt_distinguishes_workspace_files_from_artifacts() {
    let line = Adapter::build_stream_input_line(
        r#"Edit D:\Work\project\src\index.ts and keep the existing behavior."#,
    )
    .unwrap();
    let decoded: Value = serde_json::from_str(&line).unwrap();
    let content = decoded["message"]["content"].as_str().unwrap();

    assert!(content.contains("workspace files, not artifacts"));
    assert!(content.contains("do not include ArtifactMetadata"));
    assert!(
        content.ends_with(r#"Edit D:\Work\project\src\index.ts and keep the existing behavior."#)
    );
}

#[test]
fn test_initialize_advertises_image_only_with_materialized_image_support() {
    let adapter = adapter_for_test();
    let response = adapter.handle_initialize(json!(1));
    assert_eq!(
        response.result.unwrap()["agentCapabilities"]["promptCapabilities"]["image"],
        true
    );
}

#[test]
fn test_image_prompt_block_is_materialized_and_referenced() {
    let root = std::env::temp_dir().join(format!("agy-acp-image-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let params = json!({
        "prompt": [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image",
                "mimeType": "image/png",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
            }
        ]
    });

    let prepared = Adapter::prepare_prompt_content(&params, &root).unwrap();
    assert!(prepared.contains("Describe this image."));
    assert!(prepared.contains("Use the view_file tool"));
    let images: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .filter_map(Result::ok)
        .collect();
    assert_eq!(images.len(), 1);
    assert_eq!(
        images[0].path().extension().and_then(|v| v.to_str()),
        Some("png")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_native_stream_agent_delta_becomes_immediate_acp_chunk_without_final_duplicate() {
    let mut state = crate::types::NativeStreamState::default();
    let active = json!({
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-1",
            "step_type": "agent_response",
            "state": "DONE",
            "text_delta": "hello"
        }
    });
    let lines =
        Adapter::translate_native_stream_event("session-1", &active.to_string(), &mut state)
            .unwrap();
    assert_eq!(lines.len(), 1);
    let update: Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(update["params"]["update"]["content"]["text"], "hello");

    let result = json!({
        "event": "result",
        "result": {
            "conversation_id": "conv-1",
            "status": "SUCCESS",
            "response": "hello"
        }
    });
    let lines =
        Adapter::translate_native_stream_event("session-1", &result.to_string(), &mut state)
            .unwrap();
    assert!(
        lines.is_empty(),
        "final result must not duplicate streamed text"
    );
    assert_eq!(state.conversation_id.as_deref(), Some("conv-1"));
}

#[test]
fn test_direct_prompt_does_not_force_candidate_handoff_output() {
    let line = Adapter::build_stream_input_line("Reply once with the requested result.").unwrap();
    let event: Value = serde_json::from_str(&line).unwrap();
    let content = event["message"]["content"].as_str().unwrap();

    assert!(content.contains("Reply once with the requested result."));
    assert!(
        !content.contains("UNTRUSTED_CANDIDATE"),
        "direct Paseo conversations must not be forced to emit worker handoff packets"
    );
    assert!(
        !content.contains("exact changed-file inventory"),
        "direct Paseo conversations must keep normal concise output"
    );
}

#[test]
fn test_worker_prompt_uses_only_live_workspace_edit_tools() {
    let direct = Adapter::build_stream_input_line("Inspect the workspace.").unwrap();
    let direct: Value = serde_json::from_str(&direct).unwrap();
    let direct_content = direct["message"]["content"].as_str().unwrap();
    assert!(direct_content.contains("write_to_file or replace_file_content"));
    assert!(!direct_content.contains("multi_replace_file_content"));

    let assignment = json!({
        "taskId": "task-tool-contract",
        "parentAgentId": "lead-1",
        "parentRole": "lead",
        "objective": "Inspect the workspace",
        "modelId": "gpt-5.6-sol",
        "modelProvider": "openai",
        "thinkingOptionId": "medium",
        "scope": ["src"],
        "exclusions": ["all other paths"],
        "verification": ["cargo test"],
        "skillSet": ["diagnosing-bugs"]
    });
    let worker =
        Adapter::with_worker_contract("Inspect the workspace.", Some(&assignment), "plan", true);
    assert!(worker.contains("write_to_file or replace_file_content"));
    assert!(!worker.contains("multi_replace_file_content"));
}

#[test]
fn test_assigned_worker_reports_to_lead_without_asking_owner_for_permission() {
    let assignment = json!({
        "taskId": "task-no-owner-prompt",
        "parentAgentId": "lead-1",
        "parentRole": "lead",
        "objective": "Implement the assigned fix",
        "modelId": "gpt-5.6-sol",
        "modelProvider": "openai",
        "thinkingOptionId": "medium",
        "scope": ["src"],
        "exclusions": ["all other paths"],
        "verification": ["cargo test"],
        "skillSet": ["diagnosing-bugs"]
    });
    let worker = Adapter::with_worker_contract(
        "Implement the assigned fix.",
        Some(&assignment),
        "plan",
        true,
    );
    assert!(worker.contains("IMPLEMENTATION_PROPOSAL"));
    assert!(worker.contains("Lead"));
    assert!(worker.contains("Do not ask the owner"));
    assert!(worker.contains("permission question"));
}

#[test]
fn test_lead_approved_assignment_allows_preflight_then_implementation() {
    let assignment = json!({
        "taskId": "task-approved-implementation",
        "parentAgentId": "lead-1",
        "parentRole": "lead",
        "objective": "Implement the assigned fix",
        "modelId": "gpt-5.6-sol",
        "modelProvider": "openai",
        "thinkingOptionId": "medium",
        "scope": ["src"],
        "exclusions": ["all other paths"],
        "verification": ["cargo test"],
        "skillSet": ["diagnosing-bugs"],
        "approval": "IMPLEMENTATION_APPROVED"
    });
    let worker = Adapter::with_worker_contract(
        "Implement the assigned fix.",
        Some(&assignment),
        "accept-edits",
        false,
    );
    assert!(worker.contains("Lead has already approved this assignment"));
    assert!(worker.contains("then implement only within the exact scope"));
    assert!(worker.contains("Do not ask the owner whether to proceed"));
}

#[test]
fn test_lead_assigned_prompt_still_requires_candidate_handoff() {
    let assignment = json!({
        "taskId": "task-1",
        "parentRole": "lead",
        "objective": "Make the bounded change",
        "scope": ["src/example.rs"],
        "verification": ["cargo test"]
    });
    let content = Adapter::with_worker_contract(
        "Complete the assigned task.",
        Some(&assignment),
        "accept-edits",
        false,
    );

    assert!(content.contains("UNTRUSTED_CANDIDATE"));
    assert!(content.contains("task-1"));
    assert!(content.contains("A valid Lead assignment is attached"));
}

#[test]
fn test_terminal_error_after_agent_text_is_structured_and_not_assistant_text() {
    let mut state = crate::types::NativeStreamState::default();
    let completed = json!({
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-structured-error",
            "step_type": "agent_response",
            "state": "DONE",
            "text_delta": "Partial answer."
        }
    });
    Adapter::translate_native_stream_event("session-1", &completed.to_string(), &mut state)
        .unwrap();

    let failed = json!({
        "event": "result",
        "result": {
            "conversation_id": "conv-structured-error",
            "status": "ERROR",
            "response": "Partial answer.",
            "error": "request failed (code 502): upstream unavailable"
        }
    });
    let lines =
        Adapter::translate_native_stream_event("session-1", &failed.to_string(), &mut state)
            .unwrap();
    let update: Value = serde_json::from_str(&lines[0]).unwrap();

    assert_eq!(
        update["params"]["update"]["sessionUpdate"],
        "session_info_update"
    );
    assert!(update["params"]["update"]["content"].is_null());
    assert_eq!(
        update["params"]["update"]["_meta"]["agy"]["display"],
        "silent"
    );
    assert_eq!(
        update["params"]["update"]["_meta"]["agy"]["kind"],
        "terminal_error"
    );
    assert_eq!(state.emitted_text, "Partial answer.");
}
#[test]
fn test_native_stream_tool_error_after_agent_text_becomes_partial_unverified_handback() {
    let mut state = crate::types::NativeStreamState::default();
    let active = json!({
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-1",
            "step_type": "agent_response",
            "state": "DONE",
            "text_delta": "The codebase is ready."
        }
    });
    Adapter::translate_native_stream_event("session-1", &active.to_string(), &mut state).unwrap();

    let result = json!({
        "event": "result",
        "result": {
            "conversation_id": "conv-1",
            "status": "ERROR",
            "response": "The codebase is ready.",
            "error": "view_file failed: pages/chat/index.vue was not found"
        }
    });
    let lines =
        Adapter::translate_native_stream_event("session-1", &result.to_string(), &mut state)
            .unwrap();

    assert_eq!(lines.len(), 1, "the tool failure must append one warning");
    let update: Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(
        update["params"]["update"]["sessionUpdate"],
        "session_info_update"
    );
    assert!(update["params"]["update"]["content"].is_null());
    assert_eq!(
        update["params"]["update"]["_meta"]["agy"]["kind"],
        "terminal_error"
    );
    assert_eq!(
        update["params"]["update"]["_meta"]["agy"]["display"],
        "silent"
    );
    assert_eq!(state.emitted_text, "The codebase is ready.");
    assert!(
        state.error.is_none(),
        "a model-visible partial-result warning must not become a fatal ACP error"
    );
}

#[test]
fn test_interrupted_stream_after_partial_output_warns_without_duplicate_recovery() {
    let mut state = crate::types::NativeStreamState::default();
    let completed = json!({
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-interrupted",
            "step_type": "agent_response",
            "state": "DONE",
            "text_delta": "Partial progress."
        }
    });
    Adapter::translate_native_stream_event("session-1", &completed.to_string(), &mut state)
        .unwrap();

    let interrupted = json!({
        "event": "result",
        "result": {
            "conversation_id": "conv-interrupted",
            "status": "ERROR",
            "response": "Partial progress.",
            "error": "The stream was interrupted. Please continue the task you were working on."
        }
    });
    let lines =
        Adapter::translate_native_stream_event("session-1", &interrupted.to_string(), &mut state)
            .unwrap();

    assert_eq!(
        lines.len(),
        1,
        "an interruption after visible text must append one warning instead of starting a duplicate recovery turn"
    );
    let update: Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(
        update["params"]["update"]["sessionUpdate"],
        "session_info_update"
    );
    assert!(update["params"]["update"]["content"].is_null());
    assert_eq!(
        update["params"]["update"]["_meta"]["agy"]["kind"],
        "terminal_error"
    );
    assert_eq!(
        update["params"]["update"]["_meta"]["agy"]["display"],
        "silent"
    );
    assert_eq!(state.emitted_text, "Partial progress.");
    assert!(state.partial_warning_emitted);
    assert_eq!(
        state.terminal_error.as_deref(),
        Some("The stream was interrupted. Please continue the task you were working on.")
    );
}

#[test]
fn test_recoverable_edit_error_after_visible_text_warns_without_duplicate_recovery() {
    let mut state = crate::types::NativeStreamState::default();
    let completed = json!({
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-1",
            "step_type": "agent_response",
            "state": "DONE",
            "text_delta": "Task completed."
        }
    });
    Adapter::translate_native_stream_event("session-1", &completed.to_string(), &mut state)
        .unwrap();

    let failed_finalization = json!({
        "event": "result",
        "result": {
            "conversation_id": "conv-1",
            "status": "ERROR",
            "response": "Task completed.",
            "error": "fallback failed: Could not successfully apply any edits: chunk 0: target content not found in file"
        }
    });
    let lines = Adapter::translate_native_stream_event(
        "session-1",
        &failed_finalization.to_string(),
        &mut state,
    )
    .unwrap();

    assert_eq!(
        lines.len(),
        1,
        "a terminal failure after visible text must not start a second assistant response"
    );
    assert!(state.partial_warning_emitted);
    assert!(state
        .terminal_error
        .as_deref()
        .unwrap()
        .contains("target content not found"));
}

#[test]
fn test_terminal_recovery_is_bounded_and_does_not_retry_infrastructure_errors() {
    let edit_error =
        "fallback failed: Could not successfully apply any edits: target content not found in file";
    let path_error = "invalid tool call error (invalid_args) failed to read file: file not found";
    let interrupted_stream =
        "The stream was interrupted. Please continue the task you were working on.";

    let edit_prompt = Adapter::recovery_prompt_for_terminal_error(edit_error, 0).unwrap();
    assert!(edit_prompt.contains("single automatic recovery attempt"));
    assert!(edit_prompt.contains("apply smaller edits"));
    assert!(Adapter::recovery_prompt_for_terminal_error(path_error, 0).is_some());
    let interrupted_prompt =
        Adapter::recovery_prompt_for_terminal_error(interrupted_stream, 0).unwrap();
    assert!(
        interrupted_prompt.contains("same language used by the preceding user-facing response"),
        "recovery must preserve the conversation language"
    );
    assert!(Adapter::recovery_prompt_for_terminal_error(edit_error, 1).is_none());
    assert!(Adapter::recovery_prompt_for_terminal_error(interrupted_stream, 1).is_none());
    assert!(
        Adapter::terminal_recovery_prompt(interrupted_stream, 0, true).is_none(),
        "visible assistant text must suppress automatic recovery to prevent duplicate responses"
    );
    assert!(Adapter::terminal_recovery_prompt(interrupted_stream, 0, false).is_some());
    assert!(
        Adapter::recovery_prompt_for_terminal_error("selected model is at capacity", 0).is_none()
    );
    assert!(Adapter::recovery_prompt_for_terminal_error(
        "permission denied in headless print mode",
        0
    )
    .is_none());
}

#[test]
fn test_timeout_recovery_prompt_is_bounded_and_requests_synthesis() {
    let prompt = Adapter::recovery_prompt_for_timeout(0).unwrap();
    assert!(prompt.contains("single automatic recovery attempt"));
    assert!(prompt.contains("Do not invoke any further tools"));
    assert!(prompt.contains("synthesize"));
    assert!(Adapter::recovery_prompt_for_timeout(1).is_none());
    assert!(Adapter::timeout_recovery_prompt(0, false).is_some());
    assert!(
        Adapter::timeout_recovery_prompt(0, true).is_none(),
        "visible assistant text must suppress timeout recovery to prevent duplicate responses"
    );
}

#[test]
#[ignore]
fn test_persist_and_restore_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-state-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };

    adapter.persist_session("sess-1", Some("conv-abc"), 7, None);
    let restored = adapter.restore_session("sess-1");
    assert_eq!(restored, Some(("conv-abc".to_string(), 7, None)));

    let missing = adapter.restore_session("sess-unknown");
    assert_eq!(missing, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_read_response_from_db() {
    let root = std::env::temp_dir().join(format!("agy-acp-sqlite-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let db_path = conv_dir.join("test-conv.db");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();

    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(11);
    inner.extend_from_slice(b"hello world");
    let mut payload = vec![0x08, 0x0F, 0xA2, 0x01, inner.len() as u8];
    payload.extend_from_slice(&inner);

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![1i64, payload],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
        rusqlite::params![2i64, vec![0x08u8, 0x0E]],
    )
    .unwrap();
    drop(conn);

    let adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: conv_dir,
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };

    let result = adapter.read_response_from_db("test-conv", -1);
    assert_eq!(result, Some(("hello world".to_string(), 2)));

    let result = adapter.read_response_from_db("test-conv", 1);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

fn prepare_auth() -> bool {
    if std::env::var("GEMINI_API_KEY")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        eprintln!("[e2e] Using GEMINI_API_KEY");
        return true;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let settings = format!("{}/.gemini/antigravity-cli/settings.json", home);
    if std::path::Path::new(&settings).exists() {
        eprintln!("[e2e] Using local auth (keyring)");
        return true;
    }
    eprintln!("SKIP: No GEMINI_API_KEY and no local auth found");
    false
}

fn release_binary_path() -> PathBuf {
    std::env::current_dir()
        .unwrap()
        .join("target")
        .join("release")
        .join(format!("agy-acp{}", std::env::consts::EXE_SUFFIX))
}

#[test]
#[ignore]
fn test_e2e_agy_acp_full_round_trip() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    use std::time::Duration;

    if !prepare_auth() {
        return;
    }

    let agy_check = Command::new("agy").arg("--help").output();
    if agy_check.is_err() || !agy_check.unwrap().status.success() {
        eprintln!("SKIP: agy not found in PATH");
        return;
    }

    let binary = release_binary_path();
    if !binary.exists() {
        panic!("Run `cargo build --release` first");
    }

    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn agy-acp");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    let mut send_and_recv = |msg: &str| -> String {
        writeln!(stdin, "{}", msg).unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        line
    };

    let resp = send_and_recv(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );
    let init: Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(init["result"]["protocolVersion"], 1);

    let resp = send_and_recv(
        r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{"modeId":"accept-edits"}}"#,
    );
    let session: Value = serde_json::from_str(&resp).unwrap();
    let session_id = session["result"]["sessionId"].as_str().unwrap();
    assert!(!session_id.is_empty());

    let prompt_msg = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"Use the ListDir tool to inspect the current working directory, then reply exactly DIRECTORY_OK."}}]}}}}"#,
        session_id
    );
    writeln!(stdin, "{}", prompt_msg).unwrap();
    stdin.flush().unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let mut got_notification = false;
    let mut response_text = String::new();
    loop {
        if std::time::Instant::now() > deadline {
            panic!("Timed out waiting for agy-acp response");
        }
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.is_empty() {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let msg: Value = serde_json::from_str(line.trim()).unwrap();
        if msg.get("method") == Some(&json!("session/update")) {
            got_notification = true;
            response_text = msg["params"]["update"]["content"]["text"]
                .as_str()
                .unwrap_or("")
                .to_string();
        }
        if msg.get("id") == Some(&json!(3)) {
            assert!(msg["error"].is_null(), "Got error: {}", msg["error"]);
            assert_eq!(msg["result"]["stopReason"], "end_turn");
            break;
        }
    }

    drop(stdin);
    let _ = child.wait();

    assert!(got_notification, "Expected session/update notification");
    let lower = response_text.to_lowercase();
    assert!(
        lower.contains("directory_ok"),
        "Expected 'DIRECTORY_OK' after ListDir, got: '{}'",
        response_text
    );
}

fn spawn_agy_acp() -> Option<(
    std::process::ChildStdin,
    std::io::BufReader<std::process::ChildStdout>,
    std::process::Child,
)> {
    use std::io::BufReader;
    use std::process::{Command, Stdio};

    if !prepare_auth() {
        return None;
    }
    let agy_check = Command::new("agy").arg("--help").output();
    if agy_check.is_err() || !agy_check.unwrap().status.success() {
        eprintln!("SKIP: agy not found in PATH");
        return None;
    }
    let binary = release_binary_path();
    if !binary.exists() {
        panic!("Run `cargo build --release` first");
    }

    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn agy-acp");
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    Some((stdin, BufReader::new(stdout), child))
}

fn send_recv(
    stdin: &mut std::process::ChildStdin,
    reader: &mut std::io::BufReader<std::process::ChildStdout>,
    msg: &str,
) -> String {
    use std::io::{BufRead, Write};
    writeln!(stdin, "{}", msg).unwrap();
    stdin.flush().unwrap();
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    line
}

fn send_prompt_wait(
    stdin: &mut std::process::ChildStdin,
    reader: &mut std::io::BufReader<std::process::ChildStdout>,
    id: u64,
    session_id: &str,
    text: &str,
) -> (Option<String>, Value) {
    use std::io::{BufRead, Write};
    use std::time::Duration;

    let msg = format!(
        r#"{{"jsonrpc":"2.0","id":{},"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"{}"}}]}}}}"#,
        id, session_id, text
    );
    writeln!(stdin, "{}", msg).unwrap();
    stdin.flush().unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let mut notification_text: Option<String> = None;
    loop {
        if std::time::Instant::now() > deadline {
            panic!("Timed out");
        }
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.is_empty() {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let msg: Value = serde_json::from_str(line.trim()).unwrap();
        if msg.get("method") == Some(&json!("session/update")) {
            notification_text = msg["params"]["update"]["content"]["text"]
                .as_str()
                .map(String::from);
        }
        if msg.get("id") == Some(&json!(id)) {
            return (notification_text, msg);
        }
    }
}

#[test]
#[ignore]
fn test_e2e_multi_turn() {
    let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
        return;
    };

    send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );

    let resp = send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#,
    );
    let session_id = serde_json::from_str::<Value>(&resp).unwrap()["result"]["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let (text1, resp1) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        3,
        &session_id,
        "Remember this word: BANANA. Reply OK.",
    );
    assert!(resp1["error"].is_null(), "Turn 1 error: {}", resp1["error"]);
    assert!(text1.is_some());

    let (text2, resp2) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        4,
        &session_id,
        "What word did I ask you to remember? Reply with just that word.",
    );
    assert!(resp2["error"].is_null(), "Turn 2 error: {}", resp2["error"]);
    let reply = text2.unwrap_or_default().to_lowercase();
    assert!(
        reply.contains("banana"),
        "Expected 'BANANA' in multi-turn reply, got: '{}'",
        reply
    );

    drop(stdin);
    let _ = child.wait();
}

#[test]
#[ignore]
fn test_e2e_session_load() {
    let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
        return;
    };

    send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );
    let resp = send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#,
    );
    let session_id = serde_json::from_str::<Value>(&resp).unwrap()["result"]["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let (_text, resp1) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        3,
        &session_id,
        "Reply with exactly: FIRST_TURN",
    );
    assert!(
        resp1["error"].is_null(),
        "First turn error: {}",
        resp1["error"]
    );

    let (text2, resp2) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        4,
        &session_id,
        "Reply with exactly one word: SECOND",
    );
    assert!(
        resp2["error"].is_null(),
        "Second turn error: {}",
        resp2["error"]
    );
    assert!(text2.is_some(), "Expected response on continued session");

    drop(stdin);
    let _ = child.wait();
}

#[test]
#[ignore]
fn test_e2e_error_paths() {
    let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
        return;
    };

    send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );

    let resp = send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":2,"method":"session/load","params":{"sessionId":"non-existent-session"}}"#,
    );
    let val: Value = serde_json::from_str(&resp).unwrap();
    assert!(
        !val["error"].is_null(),
        "Expected error for unknown session"
    );

    let resp = send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":3,"method":"bogus/method","params":{}}"#,
    );
    let val: Value = serde_json::from_str(&resp).unwrap();
    assert!(!val["error"].is_null(), "Expected error for unknown method");

    drop(stdin);
    let _ = child.wait();
}

#[test]
#[ignore]
fn test_read_response_multi_step_no_skip_no_duplicate() {
    let root = std::env::temp_dir().join(format!("agy-acp-multi-step-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let db_path = conv_dir.join("multi.db");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();

    fn make_payload(text: &str) -> Vec<u8> {
        let text_bytes = text.as_bytes();
        let mut inner = vec![0x0A];
        let mut len = text_bytes.len();
        loop {
            if len < 128 {
                inner.push(len as u8);
                break;
            }
            inner.push((len as u8 & 0x7F) | 0x80);
            len >>= 7;
        }
        inner.extend_from_slice(text_bytes);

        let mut outer = vec![0xA2, 0x01];
        let mut ilen = inner.len();
        loop {
            if ilen < 128 {
                outer.push(ilen as u8);
                break;
            }
            outer.push((ilen as u8 & 0x7F) | 0x80);
            ilen >>= 7;
        }
        outer.extend(inner);
        outer
    }

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (1, 0, X'0801')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![2i64, make_payload("hello")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (3, 0, X'0802')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![4i64, make_payload("world")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![5i64, make_payload("line1\nline2\nline3")],
    )
    .unwrap();
    drop(conn);

    let adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: conv_dir,
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };

    let result = adapter.read_response_from_db("multi", -1);
    assert_eq!(
        result,
        Some(("hello\nworld\nline1\nline2\nline3".to_string(), 5))
    );

    let result = adapter.read_response_from_db("multi", 2);
    assert_eq!(result, Some(("world\nline1\nline2\nline3".to_string(), 5)));

    let result = adapter.read_response_from_db("multi", 4);
    assert_eq!(result, Some(("line1\nline2\nline3".to_string(), 5)));

    let result = adapter.read_response_from_db("multi", 5);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_read_response_missing_steps_table() {
    let root = std::env::temp_dir().join(format!("agy-acp-noschema-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let db_path = conv_dir.join("empty.db");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch("CREATE TABLE other (id INTEGER)")
        .unwrap();
    drop(conn);

    let adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: conv_dir,
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };

    let result = adapter.read_response_from_db("empty", -1);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_is_narration_true() {
    assert!(Adapter::is_narration("I will fetch the latest commits."));
    assert!(Adapter::is_narration("I'll fetch the latest commits."));
    assert!(Adapter::is_narration("I’ll fetch the latest commits."));
    assert!(Adapter::is_narration(
        "I will fetch the latest commits.\nI'll check the diff."
    ));
    assert!(Adapter::is_narration(
        "I will read the file.\n\nI will analyze the output."
    ));
}

#[test]
fn test_is_narration_false() {
    assert!(!Adapter::is_narration("Here is the result."));
    assert!(!Adapter::is_narration(
        "I will fetch the commits.\nHere is the result."
    ));
    assert!(!Adapter::is_narration(""));
}

#[test]
fn test_filter_narration_drops_all_narration() {
    let parts = vec![
        "I will fetch the latest commits.\nI will check the diff.".to_string(),
        "I will read the file.".to_string(),
        "The fix is confirmed! LGTM ✅".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result.as_deref(), Some("The fix is confirmed! LGTM ✅"));
}

#[test]
fn test_filter_narration_preserves_content_after_first_non_narration() {
    let parts = vec![
        "I will check things.".to_string(),
        "Here is my analysis.".to_string(),
        "I will also note this is fine.".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result.as_deref(), Some("Here is my analysis."));
}

#[test]
fn test_filter_narration_single_part_unchanged() {
    let parts = vec!["I will do something.".to_string()];
    let result = Adapter::filter_narration(&parts);
    assert_eq!(result, None);
}

#[test]
fn test_filter_narration_all_narration_drops_all() {
    let parts = vec![
        "I will fetch the file.".to_string(),
        "I'll check the output.".to_string(),
        "I will verify the fix.".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result, None);
}

#[test]
fn test_session_new_returns_models() {
    let mut adapter = adapter_for_test();
    let response = adapter.handle_session_new(json!(1), &json!({}));
    let result = response.result.as_ref().unwrap();
    assert!(result.get("sessionId").is_some());
    let models = result.get("models").unwrap();
    assert!(models.get("currentModelId").is_some());
    assert!(models.get("availableModels").is_some());
    let config_options = result.get("configOptions").unwrap().as_array().unwrap();
    assert_eq!(config_options.len(), 2);
    assert_eq!(config_options[0]["id"].as_str(), Some("model"));
    assert_eq!(config_options[0]["category"].as_str(), Some("model"));
    assert_eq!(config_options[0]["type"].as_str(), Some("select"));
    assert!(config_options[0].get("currentValue").is_some());
    assert!(config_options[0].get("options").is_some());
}

#[test]
fn test_session_new_preserves_caller_context_and_empty_mcp_servers() {
    let root = std::env::temp_dir().join(format!("agy-acp-session-context-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let cwd = root.to_string_lossy().to_string();
    let mut adapter = adapter_for_test_root(&root);
    let params = json!({
        "cwd": cwd,
        "callerContext": {"requestId": "caller-42", "transport": "stdio"},
        "clientCapabilities": {"fs": true},
        "mcpServers": [],
        "modeId": "plan"
    });

    let response = adapter.handle_session_new(json!(1), &params);
    assert!(response.error.is_none(), "error: {:?}", response.error);
    let session_id = response.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    let session = adapter.sessions.get(&session_id).unwrap();
    assert_eq!(session.caller_context, params);
    assert_eq!(session.mcp_servers, json!([]));
    assert_eq!(session.mode_id, "plan");

    let stored = adapter.load_store();
    assert_eq!(stored.sessions[&session_id].caller_context, params);
    assert_eq!(stored.sessions[&session_id].mcp_servers, json!([]));
    let args = adapter.build_agy_args(&session_id, &root.join("run.log"), None);
    assert_eq!(args[0], "--log-file");
    assert_eq!(args[1], root.join("run.log").to_string_lossy());
    assert!(!args
        .windows(2)
        .any(|pair| pair == ["--add-dir", cwd.as_str()]));
    assert!(args.windows(2).any(|pair| pair == ["--mode", "plan"]));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_session_new_rejects_required_mcp_without_echoing_configuration() {
    let root = std::env::temp_dir().join(format!("agy-acp-session-mcp-new-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut adapter = adapter_for_test_root(&root);
    let secret = "stage2-mcp-secret-must-not-escape";
    let params = json!({
        "cwd": root.to_string_lossy(),
        "mcpServers": [{
            "name": "private-api",
            "command": "private-mcp",
            "env": {"API_TOKEN": secret}
        }]
    });

    let response = adapter.handle_session_new(json!(7), &params);
    let serialized = serde_json::to_string(&response).unwrap();
    assert_eq!(response.error.as_ref().unwrap()["code"], -32001);
    assert!(serialized.contains("could not be staged securely"));
    assert!(!serialized.contains(secret));
    assert!(!serialized.contains("private-mcp"));
    assert!(adapter.sessions.is_empty());
    assert!(!adapter.state_file.exists());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_session_new_accepts_mcp_when_allowed() {
    let root = std::env::temp_dir().join(format!("agy-acp-mcp-allowed-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut adapter = adapter_for_test_root(&root);
    adapter.allow_mcp = true;
    let params = json!({
        "cwd": root.to_string_lossy(),
        "mcpServers": [{
            "name": "paseo-agents",
            "url": "http://127.0.0.1:6767/mcp/agents?callerAgentId=agy-parent"
        }]
    });

    let response = adapter.handle_session_new(json!(7), &params);
    assert!(response.error.is_none());
    assert!(response.result.is_some());
    assert_eq!(adapter.sessions.len(), 1);

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_session_load_and_resume_preserve_context_mode_and_mcp_state() {
    let root = std::env::temp_dir().join(format!("agy-acp-session-roundtrip-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let cwd = root.to_string_lossy().to_string();
    let mut first = adapter_for_test_root(&root);
    let new_params = json!({
        "cwd": cwd,
        "callerContext": {"origin": "lead", "requestId": "req-9"},
        "mcpServers": [],
        "modeId": "plan"
    });
    let session_id = first
        .handle_session_new(json!(1), &new_params)
        .result
        .unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let mut second = adapter_for_test_root(&root);
    let load_params = json!({
        "sessionId": session_id,
        "callerContext": {"loadReason": "reconnect"},
        "mcpServers": []
    });
    let load_output = second.handle_session_load(json!(2), &load_params);
    let load_response: Value = serde_json::from_str(load_output.last().unwrap()).unwrap();
    assert!(load_response["error"].is_null(), "error: {load_response:?}");
    let loaded = second.sessions.get(&session_id).unwrap();
    assert_eq!(loaded.caller_context["cwd"], cwd);
    assert_eq!(loaded.caller_context["callerContext"]["origin"], "lead");
    assert_eq!(
        loaded.caller_context["callerContext"]["loadReason"],
        "reconnect"
    );
    assert_eq!(loaded.mcp_servers, json!([]));
    assert_eq!(loaded.mode_id, "plan");

    second.sessions.clear();
    let resume = second.handle_session_resume(json!(3), &json!({"sessionId": session_id}));
    assert!(resume.error.is_none(), "error: {:?}", resume.error);
    let resumed = second.sessions.get(&session_id).unwrap();
    assert_eq!(resumed.caller_context["cwd"], cwd);
    assert_eq!(
        resumed.caller_context["callerContext"]["loadReason"],
        "reconnect"
    );
    assert_eq!(resumed.mcp_servers, json!([]));
    assert_eq!(resumed.mode_id, "plan");
    assert_eq!(
        resume.result.as_ref().unwrap()["configOptions"][1]["currentValue"],
        "plan"
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_session_load_and_resume_reject_required_mcp_without_mutating_context() {
    let root = std::env::temp_dir().join(format!("agy-acp-session-mcp-load-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut adapter = adapter_for_test_root(&root);
    let session_id = adapter
        .handle_session_new(json!(1), &json!({"callerContext": {"safe": true}}))
        .result
        .unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    let before = adapter.sessions[&session_id].clone();
    let secret = "stage2-load-mcp-secret";
    let params = json!({
        "sessionId": session_id,
        "mcpServers": {"private": {"headers": {"Authorization": secret}}}
    });

    let load_output = adapter.handle_session_load(json!(2), &params);
    let load_response: Value = serde_json::from_str(load_output.first().unwrap()).unwrap();
    assert_eq!(load_response["error"]["code"], -32001);
    assert!(!load_output.join("\n").contains(secret));
    assert_eq!(adapter.sessions[&session_id], before);

    let resume = adapter.handle_session_resume(json!(3), &params);
    let serialized = serde_json::to_string(&resume).unwrap();
    assert_eq!(resume.error.as_ref().unwrap()["code"], -32001);
    assert!(!serialized.contains(secret));
    assert_eq!(adapter.sessions[&session_id], before);

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_persisted_required_mcp_is_rejected_closed_on_restore() {
    let root = std::env::temp_dir().join(format!("agy-acp-session-mcp-store-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let adapter = adapter_for_test_root(&root);
    let secret = "stage2-persisted-mcp-secret";
    let stored = json!({
        "sessions": {
            "stored-with-mcp": {
                "conversation_id": null,
                "last_step_idx": -1,
                "model_id": null,
                "mode_id": "accept-edits",
                "caller_context": {"mcpServers": [{"env": {"TOKEN": secret}}]},
                "mcp_servers": []
            }
        }
    });
    fs::write(&adapter.state_file, serde_json::to_vec(&stored).unwrap()).unwrap();

    let mut adapter = adapter;
    let output = adapter.handle_session_load(json!(1), &json!({"sessionId": "stored-with-mcp"}));
    let response: Value = serde_json::from_str(output.first().unwrap()).unwrap();
    assert_eq!(response["error"]["code"], -32001);
    assert!(!output.join("\n").contains(secret));
    assert!(adapter.sessions.is_empty());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_delegated_restore_downgrades_persisted_full_access_to_plan() {
    let root = std::env::temp_dir().join(format!("agy-acp-stored-full-access-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let adapter = adapter_for_test_root(&root);
    let stored = json!({
        "sessions": {
            "stored-full-access": {
                "conversation_id": null,
                "last_step_idx": -1,
                "model_id": null,
                "mode_id": "full-access",
                "caller_context": {},
                "mcp_servers": []
            }
        }
    });
    fs::write(&adapter.state_file, serde_json::to_vec(&stored).unwrap()).unwrap();

    let mut adapter = adapter;
    let load = adapter.handle_session_load(json!(1), &json!({"sessionId": "stored-full-access"}));
    let load_response: Value = serde_json::from_str(load.first().unwrap()).unwrap();
    assert!(
        load_response["error"].is_null(),
        "legacy history must remain loadable"
    );
    assert_eq!(
        adapter.sessions["stored-full-access"].mode_id,
        Adapter::PLAN_MODE_ID
    );
    let resume =
        adapter.handle_session_resume(json!(2), &json!({"sessionId": "stored-full-access"}));
    assert!(
        resume.error.is_none(),
        "legacy history resume must remain loadable"
    );
    assert_eq!(
        adapter.sessions["stored-full-access"].mode_id,
        Adapter::PLAN_MODE_ID
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_session_new_exposes_default_plan_mode() {
    let mut adapter = adapter_for_test();
    let response = adapter.handle_session_new(json!(1), &json!({}));
    let config_options = response.result.as_ref().unwrap()["configOptions"]
        .as_array()
        .unwrap();
    let mode = config_options
        .iter()
        .find(|option| option["category"] == "mode")
        .expect("session/new must expose the dynamic ACP mode category");

    assert_eq!(mode["id"], "mode");
    assert_eq!(mode["currentValue"], "plan");
    assert_eq!(
        mode["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|option| option["value"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["accept-edits", "plan"]
    );
}

#[test]
fn test_full_access_mode_maps_to_accept_edits_plus_exactly_one_permission_bypass() {
    let root = std::env::temp_dir().join(format!("agy-acp-full-access-argv-{}", Uuid::new_v4()));
    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: true,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.sessions.insert(
        "full-access-session".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
            mode_id: "full-access".to_string(),
            caller_context: json!({}),
            mcp_servers: json!([]),
        },
    );

    let args = adapter.build_agy_args("full-access-session", &root.join("run.log"), None);
    let mode_index = args.iter().position(|arg| arg == "--mode").unwrap();
    assert_eq!(args[mode_index + 1], "accept-edits");
    assert_eq!(args.iter().filter(|arg| *arg == "--mode").count(), 1);
    assert_eq!(
        args.iter()
            .filter(|arg| *arg == "--dangerously-skip-permissions")
            .count(),
        1
    );
}

#[test]
fn test_plan_is_the_safe_default_and_does_not_bypass_permissions() {
    let root = std::env::temp_dir().join(format!("agy-acp-plan-default-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut adapter = adapter_for_test_root(&root);
    let response = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = response.result.unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    let args = adapter.build_agy_args(&session_id, &root.join("run.log"), None);
    assert!(args.windows(2).any(|pair| pair == ["--mode", "plan"]));
    assert!(!args
        .iter()
        .any(|arg| arg == "--dangerously-skip-permissions"));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_delegated_session_cannot_request_full_access() {
    let root = std::env::temp_dir().join(format!("agy-acp-full-access-param-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut adapter = adapter_for_test_root(&root);
    let response = adapter.handle_session_new(json!(1), &json!({"modeId": "full-access"}));
    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
    assert!(adapter.sessions.is_empty());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_autonomous_root_capability_allows_explicit_full_access() {
    let root = std::env::temp_dir().join(format!("agy-acp-root-capability-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut adapter = adapter_for_test_root(&root);
    adapter.allow_full_access = true;

    let response = adapter.handle_session_new(json!(1), &json!({"modeId": "full-access"}));
    assert!(response.error.is_none(), "error: {:?}", response.error);
    let session_id = response.result.unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    let args = adapter.build_agy_args(&session_id, &root.join("run.log"), None);
    assert!(args
        .windows(2)
        .any(|pair| pair == ["--mode", "accept-edits"]));
    assert_eq!(
        args.iter()
            .filter(|arg| *arg == "--dangerously-skip-permissions")
            .count(),
        1
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_delegated_set_mode_rejects_full_access_without_mutation() {
    let mut adapter = adapter_for_test();
    let created = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = created.result.unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let response = adapter.handle_session_set_mode(
        json!(2),
        &json!({"sessionId": session_id, "modeId": "full-access"}),
    );

    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
    assert_eq!(adapter.sessions[&session_id].mode_id, "plan");
}

#[test]
fn test_session_set_config_option_sets_plan_mode() {
    let mut adapter = adapter_for_test();
    let new_response = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = new_response.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let set_response = adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "mode", "value": "plan"}),
    );

    assert!(
        set_response.error.is_none(),
        "error: {:?}",
        set_response.error
    );
    let mode = set_response.result.as_ref().unwrap()["configOptions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|option| option["category"] == "mode")
        .unwrap();
    assert_eq!(mode["currentValue"], "plan");
}

#[test]
fn test_paseo_direct_session_set_mode_reaches_native_accept_edits() {
    let mut adapter = adapter_for_test();
    let new_response = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = new_response.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let response = adapter.handle_session_set_mode(
        json!(2),
        &json!({"sessionId": session_id, "modeId": "accept-edits"}),
    );

    assert!(response.error.is_none(), "error: {:?}", response.error);
    assert_eq!(adapter.sessions[&session_id].mode_id, "accept-edits");
    assert_eq!(adapter.agy_auto_approve_value(&session_id), "1");
    let args = adapter.build_agy_args(&session_id, &PathBuf::from("run.log"), None);
    assert!(
        args.windows(2)
            .any(|pair| pair == ["--mode", "accept-edits"]),
        "Paseo-selected accept-edits must reach the native Agy process: {args:?}"
    );
    assert_eq!(
        args.iter()
            .filter(|arg| *arg == "--dangerously-skip-permissions")
            .count(),
        1,
        "Agy headless mode requires exactly one auto-confirm flag for tool use"
    );
}

#[test]
fn direct_selected_mode_survives_session_restart() {
    for mode in [Adapter::PLAN_MODE_ID, Adapter::ACCEPT_EDITS_MODE_ID] {
        for replay in [false, true] {
            let root =
                std::env::temp_dir().join(format!("agy-acp-mode-restore-{}", Uuid::new_v4()));
            fs::create_dir_all(&root).unwrap();
            let mut adapter = adapter_for_test_root(&root);
            let created = adapter.handle_session_new(json!(1), &json!({}));
            let session_id = created.result.unwrap()["sessionId"]
                .as_str()
                .unwrap()
                .to_string();
            let selected = adapter.handle_session_set_mode(
                json!(2),
                &json!({"sessionId": session_id, "modeId": mode}),
            );
            assert!(selected.error.is_none());
            drop(adapter);

            let mut restored = adapter_for_test_root(&root);
            let request = json!({"sessionId": session_id});
            let response = if replay {
                let lines = restored.handle_session_load(json!(3), &request);
                serde_json::from_str::<Value>(&lines[0]).unwrap()
            } else {
                serde_json::to_value(restored.handle_session_resume(json!(3), &request)).unwrap()
            };
            assert!(response["error"].is_null(), "{response}");
            assert_eq!(restored.sessions[&session_id].mode_id, mode);
            let args = restored.build_agy_args(&session_id, &root.join("run.log"), None);
            assert!(args.windows(2).any(|pair| pair == ["--mode", mode]));
            assert_eq!(
                args.iter()
                    .any(|arg| arg == "--dangerously-skip-permissions"),
                mode == Adapter::ACCEPT_EDITS_MODE_ID,
            );
            drop(restored);
            fs::remove_dir_all(root).unwrap();
        }
    }
}

#[test]
fn delegated_mode_requires_approval_after_session_restart() {
    for approved in [false, true] {
        for replay in [false, true] {
            let root =
                std::env::temp_dir().join(format!("agy-acp-delegated-restore-{}", Uuid::new_v4()));
            fs::create_dir_all(&root).unwrap();
            let mut adapter = adapter_for_test_root(&root);
            let mut assignment = json!({
                "taskId": "restore-task",
                "parentRole": "lead",
                "objective": "Check the assigned workspace",
                "scope": ["src"],
                "verification": ["cargo test"]
            });
            if approved {
                assignment["approval"] = json!("IMPLEMENTATION_APPROVED");
            }
            let created =
                adapter.handle_session_new(json!(1), &json!({"agyAssignment": assignment}));
            assert!(created.error.is_none(), "{:?}", created.error);
            let session_id = created.result.unwrap()["sessionId"]
                .as_str()
                .unwrap()
                .to_string();
            let selected = adapter.handle_session_set_mode(
                json!(2),
                &json!({"sessionId": session_id, "modeId": Adapter::ACCEPT_EDITS_MODE_ID}),
            );
            assert_eq!(selected.error.is_none(), approved);
            drop(adapter);

            let mut restored = adapter_for_test_root(&root);
            let request = json!({"sessionId": session_id});
            let response = if replay {
                let lines = restored.handle_session_load(json!(3), &request);
                serde_json::from_str::<Value>(&lines[0]).unwrap()
            } else {
                serde_json::to_value(restored.handle_session_resume(json!(3), &request)).unwrap()
            };
            assert!(response["error"].is_null(), "{response}");
            let mode = if approved {
                Adapter::ACCEPT_EDITS_MODE_ID
            } else {
                Adapter::PLAN_MODE_ID
            };
            assert_eq!(restored.sessions[&session_id].mode_id, mode);
            let args = restored.build_agy_args(&session_id, &root.join("run.log"), None);
            assert!(args.windows(2).any(|pair| pair == ["--mode", mode]));
            assert_eq!(
                args.iter()
                    .any(|arg| arg == "--dangerously-skip-permissions"),
                approved
            );
            drop(restored);
            fs::remove_dir_all(root).unwrap();
        }
    }
}

#[test]
fn test_session_set_config_option_rejects_unknown_mode() {
    let mut adapter = adapter_for_test();
    let new_response = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = new_response.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap();

    let response = adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "mode", "value": "dangerous"}),
    );

    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
    assert_eq!(adapter.sessions.get(session_id).unwrap().mode_id, "plan");
}

#[test]
fn test_legacy_stored_session_deserializes_with_plan_mode() {
    let stored: crate::types::StoredSession = serde_json::from_value(json!({
        "conversation_id": "legacy-conversation",
        "last_step_idx": 7,
        "model_id": "Gemini 3.7 Flash (High)"
    }))
    .unwrap();
    let serialized = serde_json::to_value(stored).unwrap();

    assert_eq!(serialized["mode_id"], "plan");
    assert_eq!(serialized["caller_context"], json!({}));
    assert_eq!(serialized["mcp_servers"], json!([]));
}

#[test]
#[ignore]
fn test_plan_mode_persists_per_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-mode-persist-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    adapter.sessions.insert(
        "mode-persist".to_string(),
        crate::types::Session {
            conversation_id: Some("conversation-mode".to_string()),
            last_step_idx: 4,
            model_id: None,
            mode_id: crate::types::default_mode_id(),
            caller_context: json!({}),
            mcp_servers: json!([]),
        },
    );

    let response = adapter.handle_session_set_config_option(
        json!(1),
        &json!({"sessionId": "mode-persist", "configId": "mode", "value": "plan"}),
    );
    assert!(response.error.is_none());
    let stored = adapter.load_store();
    assert_eq!(stored.sessions["mode-persist"].mode_id, "plan");

    adapter.sessions.clear();
    assert!(adapter.restore_session_state("mode-persist"));
    assert_eq!(adapter.sessions["mode-persist"].mode_id, "plan");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_session_set_model() {
    let mut adapter = adapter_for_test();
    let new_resp = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let set_resp = adapter.handle_session_set_model(
        json!(2),
        &json!({"sessionId": session_id, "modelId": "Gemini 3.5 Flash (High)"}),
    );
    assert!(set_resp.error.is_none());
    assert_eq!(
        adapter
            .sessions
            .get(&session_id)
            .unwrap()
            .model_id
            .as_deref(),
        Some("Gemini 3.5 Flash (High)")
    );
}

#[test]
fn test_session_set_model_missing_params() {
    let mut adapter = adapter_for_test();
    let resp = adapter.handle_session_set_model(json!(1), &json!({}));
    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32602));
}

#[test]
fn test_session_set_model_unknown_session() {
    let mut adapter = adapter_for_test();
    let resp = adapter.handle_session_set_model(
        json!(1),
        &json!({"sessionId": "nonexistent", "modelId": "some-model"}),
    );
    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32000));
}

#[test]
fn test_session_set_config_option_sets_model() {
    let mut adapter = adapter_for_test();
    adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
    let new_resp = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let set_resp = adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "model", "value": "Model B"}),
    );

    assert!(set_resp.error.is_none(), "error: {:?}", set_resp.error);
    assert_eq!(
        adapter
            .sessions
            .get(&session_id)
            .unwrap()
            .model_id
            .as_deref(),
        Some("Model B")
    );
    let config_options = set_resp.result.as_ref().unwrap()["configOptions"]
        .as_array()
        .unwrap();
    assert_eq!(config_options[0]["currentValue"].as_str(), Some("Model B"));
}

#[test]
fn test_session_set_config_option_rejects_unknown_config() {
    let mut adapter = adapter_for_test();
    let new_resp = adapter.handle_session_new(json!(1), &json!({}));
    let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "not-model", "value": "Model B"}),
    );

    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32602));
}

#[test]
#[ignore]
fn test_session_set_model_persists() {
    let root = std::env::temp_dir().join(format!("agy-acp-model-persist-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };

    adapter.persist_session("sess-m1", Some("conv-m1"), 0, None);

    adapter.restore_session_state("sess-m1");
    adapter.handle_session_set_model(
        json!(1),
        &json!({"sessionId": "sess-m1", "modelId": "Claude Opus 4.6 (Thinking)"}),
    );

    let adapter2 = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
        allow_full_access: false,
        auto_approve: false,
        allow_mcp: false,
        model_cache: crate::adapter::ModelCache::current(),
        mcp_registry: crate::mcp::McpRegistry::new(),
    };
    let restored = adapter2.restore_session("sess-m1");
    assert_eq!(
        restored,
        Some((
            "conv-m1".to_string(),
            0,
            Some("Claude Opus 4.6 (Thinking)".to_string())
        ))
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_session_load_returns_models() {
    let mut adapter = adapter_for_test();
    adapter.sessions.insert(
        "test-load".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: Some("Gemini 3.1 Pro (High)".to_string()),
            mode_id: crate::types::default_mode_id(),
            caller_context: json!({}),
            mcp_servers: json!([]),
        },
    );
    adapter.persist_session(
        "test-load",
        Some("conv-load"),
        -1,
        Some("Gemini 3.1 Pro (High)"),
    );
    adapter.sessions.clear();

    let output = adapter.handle_session_load(json!(1), &json!({"sessionId": "test-load"}));
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(
        response["error"].is_null(),
        "error: {:?}",
        response["error"]
    );
    let models = response["result"]["models"].as_object().unwrap();
    assert_eq!(
        models["currentModelId"].as_str(),
        Some("Gemini 3.1 Pro (High)")
    );
    assert_eq!(
        response["result"]["configOptions"][0]["currentValue"].as_str(),
        Some("Gemini 3.1 Pro (High)")
    );
}

#[test]
fn test_session_resume_returns_models() {
    let mut adapter = adapter_for_test();
    adapter.persist_session(
        "test-resume",
        Some("conv-resume"),
        -1,
        Some("GPT-OSS 120B (Medium)"),
    );
    adapter.sessions.clear();

    let response = adapter.handle_session_resume(json!(1), &json!({"sessionId": "test-resume"}));
    assert!(response.error.is_none(), "error: {:?}", response.error);
    let models = response.result.as_ref().unwrap()["models"]
        .as_object()
        .unwrap();
    assert_eq!(
        models["currentModelId"].as_str(),
        Some("GPT-OSS 120B (Medium)")
    );
    assert_eq!(
        response.result.as_ref().unwrap()["configOptions"][0]["currentValue"].as_str(),
        Some("GPT-OSS 120B (Medium)")
    );
}

#[test]
fn test_session_models_json_default() {
    let mut adapter = adapter_for_test();
    let models = adapter.session_models_json(None);
    let current = models["currentModelId"].as_str().unwrap();
    if adapter.available_models.is_empty() {
        assert_eq!(current, "");
    } else {
        assert_eq!(current, adapter.available_models[0]);
    }
}

#[test]
fn test_session_models_json_with_model() {
    let mut adapter = adapter_for_test();
    adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
    let models = adapter.session_models_json(Some("Model B"));
    assert_eq!(models["currentModelId"].as_str(), Some("Model B"));
    let available = models["availableModels"].as_array().unwrap();
    assert_eq!(available.len(), 2);
    assert_eq!(available[0]["modelId"].as_str(), Some("Model A"));
    assert_eq!(available[1]["modelId"].as_str(), Some("Model B"));
}

#[test]
fn test_session_config_options_json_with_model() {
    let mut adapter = adapter_for_test();
    adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
    let config_options = adapter.session_config_options_json(Some("Model B"));
    assert_eq!(config_options[0]["id"].as_str(), Some("model"));
    assert_eq!(config_options[0]["category"].as_str(), Some("model"));
    assert_eq!(config_options[0]["type"].as_str(), Some("select"));
    assert_eq!(config_options[0]["currentValue"].as_str(), Some("Model B"));
    let options = config_options[0]["options"].as_array().unwrap();
    assert_eq!(options.len(), 2);
    assert_eq!(options[0]["value"].as_str(), Some("Model A"));
    assert_eq!(options[1]["value"].as_str(), Some("Model B"));
}

#[test]
fn test_normalize_model_id_strips_composite_tabs() {
    assert_eq!(
        Adapter::normalize_model_id("gemini-3.7-flash-high\tGemini 3.7 Flash (High)"),
        "Gemini 3.7 Flash (High)"
    );
    assert_eq!(
        Adapter::normalize_model_id("gemini-3.6-flash-high\tGemini 3.6 Flash (High)"),
        "Gemini 3.6 Flash (High)"
    );
    assert_eq!(
        Adapter::normalize_model_id("Gemini 3.7 Flash (High)"),
        "Gemini 3.7 Flash (High)"
    );
    assert_eq!(
        Adapter::normalize_model_id("gemini-3.7-flash-high"),
        "gemini-3.7-flash-high"
    );
}

#[test]
fn supervisor_checkpoint_rejects_cross_workspace_context() {
    let checkpoint = json!({
        "project": {
            "workspacePath": "D:\\Work\\FE\\TypeScript\\nuxt-js\\knm-homepage-crm-nuxt"
        }
    });
    assert!(!Adapter::checkpoint_matches_workspace(
        &checkpoint,
        "D:/Work/NX/Typescripts/vcx-scoring-admin-console/"
    ));
    assert!(Adapter::checkpoint_matches_workspace(
        &checkpoint,
        "d:/work/fe/typescript/nuxt-js/knm-homepage-crm-nuxt"
    ));
}

#[test]
fn resolve_model_alias_accepts_slugs_and_upgrades_retired_labels() {
    assert_eq!(
        Adapter::resolve_static_model_alias("gemini-4-argon-high"),
        "Gemini 4 Argon (High)"
    );
    assert_eq!(
        Adapter::resolve_static_model_alias("claude-sonnet-4-6-thinking"),
        "ClaudeSonnet 4.6 (Thinking)"
    );
    let mut adapter = adapter_for_test();
    adapter.available_models = vec![
        "Gemini 4 Argon (High)".to_string(),
        "Gemini 4 Argon (Low)".to_string(),
        "ClaudeSonnet 4.6 (Thinking)".to_string(),
    ];
    assert_eq!(
        adapter.resolve_session_model_id("gemini-4-argon-high"),
        "Gemini 4 Argon (High)"
    );
    assert_eq!(
        adapter.resolve_session_model_id("Gemini 3.8 Flash (High)"),
        "Gemini 4 Argon (High)"
    );
    assert_eq!(
        adapter.resolve_session_model_id("Gemini 3.7 Flash (Low)"),
        "Gemini 4 Argon (Low)"
    );
    assert_eq!(
        adapter.resolve_session_model_id("Claude Sonnet 4.6 (Thinking)"),
        "ClaudeSonnet 4.6 (Thinking)"
    );
}

#[test]
fn worker_assignment_accepts_optional_metadata_and_compact_followup_approval() {
    let mut adapter = adapter_for_test();
    adapter.available_models = vec!["Gemini 4 Argon (High)".to_string()];
    let new_resp = adapter.handle_session_new(
        json!(1),
        &json!({
            "cwd": "D:\\agent_workflow",
            "assignment": {
                "taskId": "task-ergonomic-1",
                "parentRole": "lead",
                "objective": "Verify compact assignment ergonomics",
                "modelId": "gemini-4-argon-high",
                "thinkingOptionId": null,
                "scope": ["src/adapter.rs"],
                "verification": ["cargo test"]
            }
        }),
    );
    assert!(
        new_resp.error.is_none(),
        "initial assignment should succeed"
    );
    let session_id = new_resp
        .result
        .as_ref()
        .and_then(|r| r.get("sessionId"))
        .and_then(Value::as_str)
        .unwrap()
        .to_string();

    let resume_resp = adapter.handle_session_resume(
        json!(2),
        &json!({
            "sessionId": session_id,
            "assignment": {
                "taskId": "task-ergonomic-1",
                "parentRole": "lead",
                "approval": "IMPLEMENTATION_APPROVED"
            }
        }),
    );
    assert!(
        resume_resp.error.is_none(),
        "compact follow-up approval should inherit initial assignment fields"
    );
}

#[tokio::test]
async fn cancelled_prompt_returns_cancelled_stop_reason_without_spawning_agy() {
    let mut adapter = adapter_for_test();
    let new_resp = adapter.handle_session_new(
        json!(1),
        &json!({
            "cwd": "D:\\agent_workflow",
            "callerContext": { "origin": "user" }
        }),
    );
    let session_id = new_resp
        .result
        .as_ref()
        .and_then(|r| r.get("sessionId"))
        .and_then(Value::as_str)
        .unwrap()
        .to_string();

    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let lines = adapter
        .handle_session_prompt(
            json!(2),
            &json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "hello" }]
            }),
            cancelled,
        )
        .await;
    assert_eq!(lines.len(), 1);
    let resp: Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(resp["result"]["stopReason"], "cancelled");
}

#[test]
fn has_slp_governance_channel_truth_table() {
    // ACP array form with an exact name match.
    assert!(Adapter::has_slp_governance_channel(&json!([
        { "name": "slp-governance", "url": "http://127.0.0.1:4000/mcp?slp=abc" }
    ])));
    // Object map form.
    assert!(Adapter::has_slp_governance_channel(&json!({
        "slp-governance": { "url": "http://127.0.0.1:4000/mcp" }
    })));
    // Null, empty array, empty object, wrong shapes, and other server names.
    assert!(!Adapter::has_slp_governance_channel(&Value::Null));
    assert!(!Adapter::has_slp_governance_channel(&json!([])));
    assert!(!Adapter::has_slp_governance_channel(&json!({})));
    assert!(!Adapter::has_slp_governance_channel(&json!([
        { "name": "paseo-slp", "url": "http://127.0.0.1:4000/mcp" }
    ])));
    assert!(!Adapter::has_slp_governance_channel(&json!([
        { "name": "slp-governance-extra" }
    ])));
    assert!(!Adapter::has_slp_governance_channel(&json!([
        "slp-governance"
    ])));
    assert!(!Adapter::has_slp_governance_channel(&json!(
        "slp-governance"
    )));
    assert!(!Adapter::has_slp_governance_channel(&json!(42)));
    // Match is found anywhere in the array.
    assert!(Adapter::has_slp_governance_channel(&json!([
        { "name": "other" },
        { "name": "slp-governance" }
    ])));
}

#[test]
fn supervisor_contract_uses_slp_mcp_checkpoint_tools_when_channel_staged() {
    let checkpoint_json =
        "{\"currentSubMission\":\"MARKER-UNIQUE-SLP-99\",\"updatedAt\":\"2026-01-01T00:00:00Z\"}";
    let content =
        Adapter::supervisor_runtime_block("Coordinate the workers.", checkpoint_json, true, None);
    assert!(content.contains("slp_checkpoint_read"));
    assert!(content.contains("slp_checkpoint_write"));
    assert!(content.contains("slp-governance"));
    assert!(!content.contains("Update-SupervisorCheckpoint.ps1"));
    // The on-disk checkpoint JSON must not be embedded; it is stale by design.
    assert!(!content.contains("MARKER-UNIQUE-SLP-99"));
    assert!(content
        .contains("read it with `slp_checkpoint_read` at turn start and after any uncertainty"));
    // Stable runtime-contract lines are unchanged.
    assert!(content.contains("<agy-acp-supervisor-runtime version=\"supervisor-v2\">"));
    assert!(content.contains("Ignore older checkpoint content when it differs"));
    assert!(content.contains("Never invent another path or replace known fields with empty values"));
    assert!(content.contains("SLP_DIRECT_PEER_CONSULTATION"));
    assert!(content.contains("Coordinate the workers."));
}

#[test]
fn supervisor_contract_skips_disk_checkpoint_read_when_channel_staged() {
    // With slp_channel=true the on-disk checkpoint context is never read or
    // embedded, so no `<supervisor-checkpoint ...>` envelope appears even for
    // a bound session.
    let content = Adapter::supervisor_contract_prompt(
        "Coordinate the workers.",
        Some("3f6c8e1a-0000-4000-8000-000000000000"),
        None,
        true,
    );
    assert!(content.contains("slp_checkpoint_read"));
    assert!(!content.contains("Update-SupervisorCheckpoint.ps1"));
    assert!(!content.contains("<supervisor-checkpoint"));
}

#[test]
fn supervisor_contract_preserves_legacy_checkpoint_io_without_channel() {
    let checkpoint_json =
        "{\"currentSubMission\":\"MARKER-UNIQUE-SLP-99\",\"updatedAt\":\"2026-01-01T00:00:00Z\"}";
    let content =
        Adapter::supervisor_runtime_block("Coordinate the workers.", checkpoint_json, false, None);
    assert!(content.contains("Update-SupervisorCheckpoint.ps1"));
    assert!(content.contains("MARKER-UNIQUE-SLP-99"));
    assert!(!content.contains("slp_checkpoint_read"));
    assert!(!content.contains("slp_checkpoint_write"));
    assert!(content.contains("Coordinate the workers."));
}

#[test]
fn worker_contract_is_unaffected_by_slp_channel_flag() {
    let assignment = json!({
        "taskId": "task-slp-flag",
        "parentRole": "lead",
        "objective": "Make the bounded change",
        "scope": ["src/example.rs"],
        "verification": ["cargo test"]
    });
    // The public wrapper hardcodes slp_channel=false for non-supervisor modes;
    // worker output must never mention the SLP checkpoint tools.
    let content = Adapter::with_worker_contract(
        "Complete the assigned task.",
        Some(&assignment),
        "accept-edits",
        false,
    );
    assert!(content.contains("UNTRUSTED_CANDIDATE"));
    assert!(content.contains("<agy-acp-worker-contract"));
    assert!(!content.contains("slp_checkpoint_read"));
    assert!(!content.contains("slp_checkpoint_write"));
    assert!(!content.contains("slp-governance"));
    assert!(!content.contains("Update-SupervisorCheckpoint.ps1"));
}

#[test]
fn supervisor_contract_renders_owner_seat_defaults_when_provided() {
    let content = Adapter::supervisor_runtime_block(
        "Coordinate the workers.",
        "",
        true,
        Some(("agy-lead", "factory-droid")),
    );
    assert!(
        content.contains("Owner-configured seat defaults: Lead `agy-lead`, Peer `factory-droid`.")
    );
    assert!(content.contains("(1) a provider the owner explicitly names in the current task"));
    assert!(content.contains("supersede any provider defaults stated elsewhere"));
    // The line rides inside the runtime envelope, before the closing tag.
    assert!(content.contains("stated in your reply.\n</agy-acp-supervisor-runtime>"));
}

#[test]
fn supervisor_contract_omits_seat_defaults_line_when_absent() {
    let content = Adapter::supervisor_runtime_block("Coordinate the workers.", "CHK", false, None);
    assert!(!content.contains("Owner-configured seat defaults"));
    // Legacy byte shape: the dispatch sentence flows straight into the tag.
    assert!(content.contains("terminal for this turn.\n</agy-acp-supervisor-runtime>"));
}

#[test]
fn seat_defaults_env_parsing_is_lenient() {
    assert_eq!(
        Adapter::parse_seat_defaults_env("{\"lead\":\"agy-lead\",\"peer\":\"factory-droid\"}"),
        Some(("agy-lead".to_string(), "factory-droid".to_string()))
    );
    assert_eq!(Adapter::parse_seat_defaults_env("not json"), None);
    assert_eq!(
        Adapter::parse_seat_defaults_env("{\"lead\":\"agy-lead\"}"),
        None
    );
    assert_eq!(
        Adapter::parse_seat_defaults_env("{\"lead\":\"\",\"peer\":\"x\"}"),
        None
    );
    assert_eq!(
        Adapter::parse_seat_defaults_env("{\"lead\":1,\"peer\":\"x\"}"),
        None
    );
}
