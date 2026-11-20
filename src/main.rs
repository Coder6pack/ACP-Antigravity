mod account;
mod adapter;
mod db;
mod mcp;
mod protobuf;
mod streaming;
mod types;

#[cfg(test)]
mod tests;

use serde_json::json;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tokio::sync::mpsc;

use adapter::Adapter;
use clap::Parser;
use types::{JsonRpcRequest, JsonRpcResponse};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Deprecated compatibility flag. Narration remains enabled so callers
    /// receive visible progress during long tool-driven turns.
    #[arg(long = "skip-naration", default_value_t = false)]
    skip_naration: bool,

    /// Process-level capability for a separate Lead seat. Supervisor and Peer
    /// launchers must never set it.
    #[arg(long = "allow-lead-full-access", default_value_t = false)]
    allow_lead_full_access: bool,

    /// Auto-approve all tool permission requests without prompting in accept-edits mode.
    #[arg(long = "auto-approve", default_value_t = false)]
    auto_approve: bool,

    /// Alias for --auto-approve matching Agy native flag.
    #[arg(long = "dangerously-skip-permissions", default_value_t = false)]
    dangerously_skip_permissions: bool,

    /// Allow non-empty MCP servers in session parameters. Coordinator launchers
    /// must grant this explicitly; Peer providers fail closed by default.
    #[arg(long = "allow-mcp", default_value_t = false)]
    allow_mcp: bool,

    /// Print an opaque fingerprint of the active Agy credential for catalog refresh.
    #[arg(long = "credential-fingerprint", default_value_t = false)]
    credential_fingerprint: bool,
}

impl Cli {
    fn effective_skip_naration(&self) -> bool {
        false
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if cli.credential_fingerprint {
        match account::active_credential_fingerprint() {
            Some(fingerprint) => println!("{fingerprint:016x}"),
            None => println!("none"),
        }
        return;
    }
    let lead_auto_approve =
        cli.allow_lead_full_access && (cli.auto_approve || cli.dangerously_skip_permissions);
    std::env::set_var(
        "AGY_AUTO_APPROVE",
        if lead_auto_approve { "1" } else { "0" },
    );
    std::env::set_var("AGY_ALLOW_MCP", if cli.allow_mcp { "1" } else { "0" });
    let adapter =
        Adapter::new_with_capabilities(cli.effective_skip_naration(), cli.allow_lead_full_access);
    let adapter = Arc::new(tokio::sync::Mutex::new(adapter));
    let active_cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let session_prompt_locks: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Option<String>>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) if !l.trim().is_empty() => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
                _ => {}
            }
        }
    });

    let mut stdout = io::stdout();
    let mut stdin_open = true;
    let mut pending_prompts = 0usize;

    loop {
        if !stdin_open && pending_prompts == 0 {
            break;
        }

        let line = if stdin_open {
            tokio::select! {
                output = out_rx.recv() => {
                    match output {
                        Some(Some(line)) => {
                            let _ = writeln!(stdout, "{}", line);
                            let _ = stdout.flush();
                        }
                        Some(None) => pending_prompts = pending_prompts.saturating_sub(1),
                        None => {}
                    }
                    continue;
                }
                input = rx.recv() => {
                    match input {
                        Some(line) => line,
                        None => {
                            stdin_open = false;
                            continue;
                        }
                    }
                }
            }
        } else {
            match out_rx.recv().await {
                Some(Some(line)) => {
                    let _ = writeln!(stdout, "{}", line);
                    let _ = stdout.flush();
                }
                Some(None) => pending_prompts = pending_prompts.saturating_sub(1),
                None => break,
            }
            continue;
        };

        while let Ok(output) = out_rx.try_recv() {
            match output {
                Some(line) => {
                    let _ = writeln!(stdout, "{}", line);
                    let _ = stdout.flush();
                }
                None => pending_prompts = pending_prompts.saturating_sub(1),
            }
        }

        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let id = match req.id {
            Some(id) => id,
            None => {
                if req.method.as_deref() == Some("session/cancel") {
                    let params = req.params.unwrap_or(json!({}));
                    if let Some(session_id) = params.get("sessionId").and_then(|v| v.as_str()) {
                        if let Some(cancelled) = active_cancellations
                            .lock()
                            .unwrap()
                            .get(session_id)
                            .cloned()
                        {
                            cancelled.store(true, Ordering::SeqCst);
                        }
                    }
                }
                continue;
            }
        };

        let output = match req.method.as_deref() {
            Some("initialize") => {
                let adapter = adapter.lock().await;
                vec![serde_json::to_string(&adapter.handle_initialize(id)).unwrap()]
            }
            Some("session/new") => {
                let params = req.params.unwrap_or(json!({}));
                let mut adapter = adapter.lock().await;
                vec![serde_json::to_string(&adapter.handle_session_new(id, &params)).unwrap()]
            }
            Some("session/load") => {
                let params = req.params.unwrap_or(json!({}));
                let mut adapter = adapter.lock().await;
                adapter.handle_session_load(id, &params)
            }
            Some("session/resume") => {
                let params = req.params.unwrap_or(json!({}));
                let mut adapter = adapter.lock().await;
                vec![serde_json::to_string(&adapter.handle_session_resume(id, &params)).unwrap()]
            }
            Some("session/prompt") => {
                let params = req.params.unwrap_or(json!({}));
                let session_id = params
                    .get("sessionId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let cancelled = Arc::new(AtomicBool::new(false));
                if !session_id.is_empty() {
                    if let Some(previous) = active_cancellations
                        .lock()
                        .unwrap()
                        .insert(session_id.clone(), Arc::clone(&cancelled))
                    {
                        previous.store(true, Ordering::SeqCst);
                    }
                }
                let session_lock = if !session_id.is_empty() {
                    Some(
                        session_prompt_locks
                            .lock()
                            .unwrap()
                            .entry(session_id.clone())
                            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                            .clone(),
                    )
                } else {
                    None
                };
                let adapter = Arc::clone(&adapter);
                let active_cancellations = Arc::clone(&active_cancellations);
                let out_tx = out_tx.clone();
                pending_prompts += 1;
                tokio::spawn(async move {
                    let _session_guard = match &session_lock {
                        Some(lock) => Some(lock.lock().await),
                        None => None,
                    };
                    let mut prompt_adapter = {
                        let adapter = adapter.lock().await;
                        adapter.clone()
                    };
                    let cancelled_for_cleanup = Arc::clone(&cancelled);
                    let output = prompt_adapter
                        .handle_session_prompt(id, &params, cancelled)
                        .await;
                    if !session_id.is_empty() {
                        let mut adapter = adapter.lock().await;
                        adapter.merge_prompt_session_from(&session_id, &prompt_adapter);
                    }
                    if !session_id.is_empty() {
                        let mut map = active_cancellations.lock().unwrap();
                        if let Some(current) = map.get(&session_id) {
                            if Arc::ptr_eq(current, &cancelled_for_cleanup) {
                                map.remove(&session_id);
                            }
                        }
                    }
                    for line in output {
                        let _ = out_tx.send(Some(line));
                    }
                    let _ = out_tx.send(None);
                });
                Vec::new()
            }
            Some("session/cancel") => {
                let params = req.params.unwrap_or(json!({}));
                if let Some(session_id) = params.get("sessionId").and_then(|v| v.as_str()) {
                    if let Some(cancelled) = active_cancellations
                        .lock()
                        .unwrap()
                        .get(session_id)
                        .cloned()
                    {
                        cancelled.store(true, Ordering::SeqCst);
                    }
                }
                let r = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({})),
                    error: None,
                };
                vec![serde_json::to_string(&r).unwrap()]
            }
            Some("session/set_model") | Some("session/setModel") => {
                let params = req.params.unwrap_or(json!({}));
                let mut adapter = adapter.lock().await;
                vec![serde_json::to_string(&adapter.handle_session_set_model(id, &params)).unwrap()]
            }
            Some("session/set_mode") | Some("session/setMode") => {
                let params = req.params.unwrap_or(json!({}));
                let mut adapter = adapter.lock().await;
                vec![serde_json::to_string(&adapter.handle_session_set_mode(id, &params)).unwrap()]
            }
            Some("session/set_config_option") | Some("session/setConfigOption") => {
                let params = req.params.unwrap_or(json!({}));
                let mut adapter = adapter.lock().await;
                vec![
                    serde_json::to_string(&adapter.handle_session_set_config_option(id, &params))
                        .unwrap(),
                ]
            }
            Some(method) => {
                let r = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(
                        json!({"code":-32601,"message":format!("method not found: {method}")}),
                    ),
                };
                vec![serde_json::to_string(&r).unwrap()]
            }
            None => continue,
        };

        for line in output {
            let _ = writeln!(stdout, "{}", line);
        }
        let _ = stdout.flush();
    }
}
