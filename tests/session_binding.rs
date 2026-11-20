#![cfg(unix)]

use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;
use uuid::Uuid;

struct Host {
    child: Child,
    input: ChildStdin,
    output: Receiver<String>,
    root: PathBuf,
}

impl Host {
    fn start(first_id: &str, replacement_id: &str) -> Self {
        let root = std::env::temp_dir().join(format!("acp-binding-{}", Uuid::new_v4()));
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let agy = bin.join("agy");
        fs::write(&agy, include_str!("fixtures/agy-conversation.sh")).unwrap();
        fs::set_permissions(&agy, fs::Permissions::from_mode(0o755)).unwrap();
        let mut paths = vec![bin];
        paths.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        let mut child = Command::new(env!("CARGO_BIN_EXE_agy-acp"))
            .env("HOME", &root)
            .env("PATH", std::env::join_paths(paths).unwrap())
            .env("AGY_MODEL_DISCOVERY_BIN", &agy)
            .env("ACP_TEST_FIRST_ID", first_id)
            .env("ACP_TEST_REPLACEMENT_ID", replacement_id)
            .current_dir(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, output) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        });
        Self {
            child,
            input,
            output,
            root,
        }
    }

    fn request(&mut self, id: u64, method: &str, params: Value) -> (Value, String) {
        writeln!(
            self.input,
            "{}",
            json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
        )
        .unwrap();
        self.input.flush().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut text = String::new();
        loop {
            let line = self
                .output
                .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                .expect("ACP request did not complete");
            let value: Value = serde_json::from_str(&line).expect("ACP output must be JSON");
            if value["id"] == id {
                assert!(value["error"].is_null(), "{value}");
                return (value["result"].clone(), text);
            }
            if value["params"]["update"]["sessionUpdate"] == "agent_message_chunk" {
                text.push_str(
                    value["params"]["update"]["content"]["text"]
                        .as_str()
                        .unwrap(),
                );
            }
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn next_prompt_uses_the_conversation_selected_by_the_previous_run() {
    let first_id = Uuid::new_v4().to_string();
    let replacement_id = Uuid::new_v4().to_string();
    let mut host = Host::start(&first_id, &replacement_id);
    host.request(1, "initialize", json!({"protocolVersion": 1}));
    let (session, _) = host.request(2, "session/new", json!({}));
    let session_id = session["sessionId"].as_str().unwrap();
    for (id, prompt, expected) in [
        (3, "start-conversation", "none"),
        (4, "replace-conversation", first_id.as_str()),
        (5, "follow-replacement", replacement_id.as_str()),
    ] {
        let (result, text) = host.request(
            id,
            "session/prompt",
            json!({
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": prompt}],
            }),
        );
        assert_eq!(result["stopReason"], "end_turn");
        assert_eq!(text, format!("received:{expected}"));
    }
}
