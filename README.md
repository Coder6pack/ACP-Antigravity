# ACP-Antigravity

An [Agent Client Protocol (ACP)](https://agentclientprotocol.com) stdio adapter for [Google Antigravity CLI](https://github.com/google-antigravity/antigravity-cli) (`agy`). It bridges `agy` into any ACP-compatible host like [Zed](https://zed.dev), enabling you to use Gemini models through `agy` inside Zed's Agent Panel.

## How It Works

`agy-acp` speaks JSON-RPC over stdin/stdout (the ACP transport). When a host like Zed sends a prompt, `agy-acp` spawns `agy` as a subprocess, gives that invocation a unique log file so its conversation ID can be bound safely under parallel execution, streams SQLite-backed updates as incremental `session/update` notifications, and persists session state across restarts. Captured `agy --print` stdout supplies the final-answer fallback. Operational prompt failures without an assistant handback are emitted as model-visible `[System Error]` assistant updates with their code and diagnostic data, followed by a valid `end_turn`; this prevents ACP hosts that stringify JSON-RPC error objects from reducing the failure to `[object Object]`. A recoverable workspace-tool drift or native `The stream was interrupted` result resumes the same bound conversation exactly once. A Google eligibility EOF or `UNAVAILABLE 503` that occurs before any agent output is retried at most twice with 1s/2s backoff; once output exists, the prompt is never replayed. If bounded recovery fails, or another tool/operation fails after assistant text was produced, the adapter preserves the partial response, emits a typed silent diagnostic in ACP session_info_update._meta, and ends the turn normally; the diagnostic is retained for telemetry without adding assistant text. Protocol validation errors remain JSON-RPC errors.

Prompt execution uses a per-request adapter snapshot. The shared adapter mutex is held only while taking that snapshot and merging the completed session, never while awaiting the `agy` subprocess. Independent ACP sessions can therefore initialize and run concurrently without one long prompt blocking every other session.

At startup the adapter runs `agy models` through `AGY_MODEL_DISCOVERY_BIN` (or `agy` on `PATH`) and preserves native order. On Windows it fingerprints the active Agy credential in memory and fetches models again when that credential changes. A failed discovery falls back only to the catalog cached for that credential. When no credential is readable, each new session fetches models again. `AGY_ACP_MODELS_JSON` is a legacy startup fallback only in that case and is never merged into a successful native result.

```
Zed (ACP host)  <--stdin/stdout JSON-RPC-->  agy-acp  <--subprocess-->  agy  <--API-->  Gemini
```

## Prerequisites

- **Current stable Rust** with Cargo and a C compiler (for bundled SQLite)
- **`agy`** installed and in your `PATH` — install from [google-antigravity/antigravity-cli releases](https://github.com/google-antigravity/antigravity-cli)
- **Authentication** — sign in to the Agy CLI on the machine that runs the adapter

## Build & Install

```bash
cargo build --release --locked
```

The binary is at `target/release/agy-acp`. Copy it somewhere in your `PATH`:

```bash
cp target/release/agy-acp /usr/local/bin/
```

## Use with Zed

Add `agy-acp` as a custom agent server in your Zed settings (`~/.config/zed/settings.json`):

```json
{
  "agent_servers": {
    "agy": {
      "type": "custom",
      "command": "agy-acp",
      "args": [],
      "env": {}
    }
  }
}
```

Then open the Agent Panel in Zed (`Cmd-?` on macOS, `Ctrl-?` on Linux), select **agy** from the agent dropdown, and start chatting.

### Model Selection

`agy-acp` queries available models by running `agy models` at startup. You can switch models from Zed's model selector in the agent thread — the adapter exposes them as ACP config options.

### Permission Mode

The adapter exposes one dynamic ACP `mode` selector. New ordinary sessions
default to `plan`. In Agy headless print mode, tools that request confirmation
(including read-only MCP calls) are denied in this mode. MCP attachment and
successful chat alone do not establish that tool execution works.

For a direct Paseo session with no delegated assignment, explicitly select
`accept-edits` in the host to authorize tool execution. The adapter persists
that selection across load/resume and starts Agy with `accept-edits` and the
native auto-confirm flag. This permits tools, including writes; it is not a
read-only MCP mode. Do not select it when only read-only authority was granted.

For a delegated assignment, a
matching Lead `IMPLEMENTATION_APPROVED` or one consolidated
`REMEDIATION_APPROVED` envelope promotes that task-bound session to
`accept-edits`; callers cannot unlock it with a mode flag alone. Ordinary
Peer processes do not advertise or accept `full-access`, including through
load/resume of persisted state. A separate `agy-lead` launcher grants that
process-level capability with `--allow-lead-full-access`; session input cannot
grant it. Legacy stored sessions without a mode remain
`plan` rather than being silently escalated. Every `agy` invocation
receives exactly one native `--mode <value>` argument. `AGY_EXTRA_ARGS` cannot
override the selected mode or inject its own permission-bypass flag. Both ACP
mode surfaces are supported: Paseo's native `session/set_mode` and
selector-based `session/set_config_option` update the same persisted adapter
mode state.

Delegated assignments without approval remain in `plan` after restore.
Selecting a mode does not replace their Lead approval requirement.

When Agy's headless print mode soft-denies a tool confirmation, the adapter
returns a visible system-error update identifying the available permission
paths, rather than reporting a generic empty-response failure.

### Per-session MCP

The installed Agy CLI exposes only user-level MCP configuration, so `agy-acp`
stages supported ACP HTTP `mcpServers` inside a seat-local Gemini config
directory instead of mutating the user's global configuration. The adapter
selects that directory with Agy's `--gemini_dir` flag and points
`--app_data_dir` back to the original authenticated conversation store. It
does not replace `HOME`, `USERPROFILE`, or XDG roots, because those variables
also control conversation continuity and replacing them can suppress the first
stream event on `--conversation` resumes. Each upstream server is reached
through a loopback proxy that retains the complete path/query (including
Paseo's `callerAgentId`) and injects configured headers without writing their
values into Agy's config file. Unsupported transports and malformed inputs
fail closed. Secret-bearing MCP definitions remain in adapter memory and are
not persisted.

`scripts/Test-AgyPaseoMcpBoundary.ps1` exercises session acceptance over the
real JSON-RPC stdio boundary. Runtime tool discovery, child creation, and
direct-parent attention remain separate live smoke gates.

### Passing Extra Arguments

Set the `AGY_EXTRA_ARGS` environment variable to pass additional arguments to every `agy` invocation:

```json
{
  "agent_servers": {
    "agy": {
      "type": "custom",
      "command": "agy-acp",
      "args": [],
      "env": {
        "AGY_EXTRA_ARGS": "--some-flag value"
      }
    }
  }
}
```

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key for Gemini (passed through to `agy`) |
| `AGY_EXTRA_ARGS` | Space-separated extra args passed to every `agy` invocation |
| `AGY_PRINT_TIMEOUT` | Maximum duration of an `agy` turn (default `1h`; omitting `--print-timeout` would restore Agy's native 5-minute default) |
| `AGY_AUTO_APPROVE` | The adapter overrides ambient values using the session's effective mode. Explicit direct `accept-edits`, Lead-approved delegated writes, and separately gated full access enable native auto-confirm; `plan` does not. |

## Session Persistence

Sessions are persisted to `~/.openab/agy-acp/sessions.json`, including the
caller context, selected mode, and empty MCP context. When you resume a session
in Zed, `agy-acp` restores the conversation binding and replays the message
history from `agy`'s SQLite conversation databases
(`~/.gemini/antigravity-cli/conversations/*.db`). Session `cwd` is used for
that session's Agy invocation; no caller context is written to global Agy
configuration.

## Debugging

To inspect the JSON-RPC messages between Zed and `agy-acp`, run `dev: open acp logs` from Zed's Command Palette.

## License

MIT
