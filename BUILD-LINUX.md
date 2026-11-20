# Build on Linux

Requires current stable Rust/Cargo and a C compiler. The official `agy` CLI must be installed and signed in on the machine that runs the adapter.

```sh
unzip agy-acp-0.1.20-r10-source.zip -d agy-acp-0.1.20-r10-source
cd agy-acp-0.1.20-r10-source
cargo build --release --locked
./target/release/agy-acp --version
agy models
```

Point your ACP host at the built `target/release/agy-acp` executable. Keep `agy` on PATH, or set `AGY_MODEL_DISCOVERY_BIN` to its absolute path.

On Linux, the Windows Credential Manager fingerprint is unavailable. The adapter fetches models at startup and at each new ACP session. If Paseo has cached its provider model list after an account switch, run `paseo reload` to refresh that UI catalog. The separate Paseo plugin's automatic credential watcher currently applies only on Windows.
