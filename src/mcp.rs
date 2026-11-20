use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use uuid::Uuid;

const MAX_HTTP_HEADERS: usize = 64 * 1024;
const MAX_REQUEST_BODY: usize = 32 * 1024 * 1024;
const MAX_MCP_SERVERS: usize = 32;
const MAX_MCP_HEADERS: usize = 64;
const MAX_CONFIGURED_HEADER_BYTES: usize = 64 * 1024;
const MAX_PROXY_CONNECTIONS: usize = 32;
const PROXY_READ_TIMEOUT: Duration = Duration::from_millis(250);
const PROXY_ACCEPT_BACKOFF: Duration = Duration::from_millis(10);
const PROXY_HEADER_TIMEOUT: Duration = Duration::from_secs(5);
const PROXY_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// The adapter accepts the ACP HTTP form and presents Agy with a local
/// streamable-HTTP endpoint.  Agy 1.1.26 persists this shape under
/// `.gemini/config/mcp_config.json`.
#[derive(Clone)]
pub(crate) struct McpAttachment {
    servers: Vec<NormalizedMcpServer>,
    /// Stable identity of non-secret attachment metadata.  Header values are
    /// deliberately excluded so this can be persisted without retaining a
    /// credential-derived value.
    pub(crate) identity: String,
}

#[derive(Clone)]
struct NormalizedMcpServer {
    name: String,
    endpoint: HttpEndpoint,
    headers: Vec<HttpHeader>,
}

impl PartialEq for NormalizedMcpServer {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.endpoint == other.endpoint && self.headers == other.headers
    }
}

#[derive(Clone, PartialEq, Eq)]
struct HttpHeader {
    name: String,
    value: String,
}

#[derive(Clone, PartialEq, Eq)]
struct HttpEndpoint {
    authority: String,
    host: String,
    port: u16,
    path_and_query: String,
}

impl McpAttachment {
    pub(crate) fn normalized_value(&self) -> Value {
        Value::Array(
            self.servers
                .iter()
                .map(|server| {
                    let headers: Vec<Value> = server
                        .headers
                        .iter()
                        .map(|header| {
                            json!({
                                "name": header.name,
                                "value": header.value,
                            })
                        })
                        .collect();
                    json!({
                        "type": "http",
                        "name": server.name,
                        "url": server.endpoint.as_url(),
                        "headers": headers,
                    })
                })
                .collect(),
        )
    }

    pub(crate) fn same_credentials(&self, other: &Self) -> bool {
        self.servers == other.servers
    }

    fn config_json(&self, local_urls: &[String]) -> Value {
        let mut servers = Map::new();
        for (server, local_url) in self.servers.iter().zip(local_urls) {
            servers.insert(
                server.name.clone(),
                json!({
                    "disabled": false,
                    "serverUrl": local_url,
                }),
            );
        }
        json!({ "mcpServers": servers })
    }

    fn server_count(&self) -> usize {
        self.servers.len()
    }
}

impl HttpEndpoint {
    fn as_url(&self) -> String {
        format!("http://{}{}", self.authority, self.path_and_query)
    }
}

/// Parse and normalize the only MCP shape this adapter can attach without
/// putting caller-controlled credentials in Agy configuration or argv.
pub(crate) fn normalize_mcp_servers(value: &Value) -> Result<Option<McpAttachment>, ()> {
    let Value::Array(entries) = value else {
        return match value {
            Value::Null => Ok(None),
            // Preserve the old empty-object tolerance, but never interpret a
            // nonempty object as a different ACP server shape.
            Value::Object(entries) if entries.is_empty() => Ok(None),
            _ => Err(()),
        };
    };

    if entries.is_empty() {
        return Ok(None);
    }
    if entries.len() > MAX_MCP_SERVERS {
        return Err(());
    }

    let mut servers = Vec::with_capacity(entries.len());
    let mut names = HashSet::with_capacity(entries.len());
    for entry in entries {
        let Some(entry) = entry.as_object() else {
            return Err(());
        };
        if entry.keys().any(|key| {
            !matches!(
                key.as_str(),
                "type" | "name" | "url" | "serverUrl" | "headers"
            )
        }) {
            return Err(());
        }

        let Some(name) = entry.get("name").and_then(Value::as_str) else {
            return Err(());
        };
        let name = name.trim();
        if !valid_server_name(name) || !names.insert(name.to_string()) {
            return Err(());
        }

        let server_type = entry.get("type").and_then(Value::as_str).unwrap_or("http");
        if !server_type.eq_ignore_ascii_case("http")
            || entry.get("command").is_some()
            || entry.get("args").is_some()
            || entry.get("env").is_some()
        {
            return Err(());
        }

        let url = match (entry.get("url"), entry.get("serverUrl")) {
            (Some(_), Some(_)) => return Err(()),
            (Some(url), None) | (None, Some(url)) => url.as_str().ok_or(())?,
            (None, None) => return Err(()),
        };
        let endpoint = parse_http_endpoint(url)?;
        let headers = parse_headers(entry.get("headers"))?;
        servers.push(NormalizedMcpServer {
            name: name.to_string(),
            endpoint,
            headers,
        });
    }

    // MCP server ordering has no protocol meaning.  Canonical ordering makes
    // a load/resume identity independent of incidental caller ordering while
    // the original secret-bearing values remain only in memory.
    servers.sort_by(|left, right| left.name.cmp(&right.name));
    let identity = attachment_identity(&servers);
    Ok(Some(McpAttachment { servers, identity }))
}

fn valid_server_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 256
        && name
            .bytes()
            .all(|byte| byte >= 0x21 && byte != b'/' && byte != b'\\' && byte != b'"')
}

fn parse_headers(value: Option<&Value>) -> Result<Vec<HttpHeader>, ()> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Some(entries) = value.as_array() else {
        return Err(());
    };
    if entries.len() > MAX_MCP_HEADERS {
        return Err(());
    }

    let mut seen = HashSet::with_capacity(entries.len());
    let mut headers = Vec::with_capacity(entries.len());
    let mut header_bytes = 0usize;
    for entry in entries {
        let Some(entry) = entry.as_object() else {
            return Err(());
        };
        let Some(name) = entry.get("name").and_then(Value::as_str) else {
            return Err(());
        };
        let Some(value) = entry.get("value").and_then(Value::as_str) else {
            return Err(());
        };
        let name = name.trim();
        if !valid_header_name(name) || !valid_header_value(value) {
            return Err(());
        }
        header_bytes = header_bytes
            .checked_add(name.len())
            .and_then(|bytes| bytes.checked_add(value.len()))
            .ok_or(())?;
        if header_bytes > MAX_CONFIGURED_HEADER_BYTES {
            return Err(());
        }
        let lower_name = name.to_ascii_lowercase();
        if !seen.insert(lower_name) {
            return Err(());
        }
        // The proxy owns these framing headers.  Letting a caller override
        // them would make forwarding ambiguous and can desynchronize the
        // local and remote HTTP streams.
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "host"
                | "connection"
                | "keep-alive"
                | "proxy-connection"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailer"
                | "transfer-encoding"
                | "upgrade"
                | "content-length"
        ) {
            return Err(());
        }
        headers.push(HttpHeader {
            name: name.to_ascii_lowercase(),
            value: value.to_string(),
        });
    }

    headers.sort_by(|left, right| {
        left.name
            .to_ascii_lowercase()
            .cmp(&right.name.to_ascii_lowercase())
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(headers)
}

fn valid_header_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 256 && name.bytes().all(is_http_token_byte)
}

fn is_http_token_byte(byte: u8) -> bool {
    matches!(
        byte,
        b'0'..=b'9'
            | b'a'..=b'z'
            | b'A'..=b'Z'
            | b'!'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'*'
            | b'+'
            | b'-'
            | b'.'
            | b'^'
            | b'_'
            | b'`'
            | b'|'
            | b'~'
    )
}

fn valid_header_value(value: &str) -> bool {
    value.len() <= 16 * 1024
        && !value
            .bytes()
            .any(|byte| byte == 0 || byte == b'\r' || byte == b'\n')
}

fn parse_http_endpoint(raw: &str) -> Result<HttpEndpoint, ()> {
    let raw = raw.trim();
    if raw.len() > 16 * 1024
        || !raw
            .get(..7)
            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("http://"))
        || raw
            .bytes()
            .any(|byte| byte == 0 || byte == b'\r' || byte == b'\n')
        || raw.contains('#')
        || raw.contains('@')
        || raw.contains('\\')
    {
        return Err(());
    }

    let rest = &raw[7..];
    let authority_end = rest.find(['/', '?']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    if authority.is_empty() || authority.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(());
    }

    let (host, port) = parse_authority(authority)?;
    let suffix = &rest[authority_end..];
    let path_and_query = if suffix.is_empty() {
        "/".to_string()
    } else if suffix.starts_with('?') {
        format!("/{suffix}")
    } else {
        suffix.to_string()
    };
    if !valid_path_and_query(&path_and_query) || !valid_query(&path_and_query) {
        return Err(());
    }

    Ok(HttpEndpoint {
        authority: authority.to_string(),
        host,
        port,
        path_and_query,
    })
}

fn parse_authority(authority: &str) -> Result<(String, u16), ()> {
    if authority.starts_with('[') {
        let close = authority.find(']').ok_or(())?;
        let host = &authority[1..close];
        if host.is_empty() {
            return Err(());
        }
        let port = if close + 1 == authority.len() {
            80
        } else {
            let port = authority.get(close + 1..).ok_or(())?;
            let port = port.strip_prefix(':').ok_or(())?;
            parse_port(port)?
        };
        return Ok((host.to_string(), port));
    }

    if authority.contains(':') {
        let (host, raw_port) = authority.rsplit_once(':').ok_or(())?;
        if host.is_empty() || host.contains(':') {
            return Err(());
        }
        return Ok((host.to_string(), parse_port(raw_port)?));
    }
    Ok((authority.to_string(), 80))
}

fn parse_port(raw: &str) -> Result<u16, ()> {
    if raw.is_empty() || raw.len() > 5 {
        return Err(());
    }
    let port = raw.parse::<u16>().map_err(|_| ())?;
    if port == 0 {
        return Err(());
    }
    Ok(port)
}

fn valid_path_and_query(path: &str) -> bool {
    !path.is_empty()
        && path.starts_with('/')
        && path.len() <= 16 * 1024
        && !path
            .bytes()
            .any(|byte| byte == 0 || byte == b'\r' || byte == b'\n' || byte == b' ')
}

fn is_paseo_agent_control_endpoint(path_and_query: &str) -> bool {
    let path = path_and_query
        .split_once('?')
        .map_or(path_and_query, |(path, _)| path);
    percent_decode_ascii(path).is_some_and(|decoded| decoded == "/mcp/agents")
}

fn valid_query(path: &str) -> bool {
    let Some(query) = path.split_once('?').map(|(_, query)| query) else {
        return true;
    };
    if query.is_empty() {
        return false;
    }
    let mut caller_agent_id_count = 0usize;
    for part in query.split('&') {
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        let Some(decoded_key) = percent_decode_ascii(key) else {
            return false;
        };
        let lower_key = decoded_key.to_ascii_lowercase();
        if matches!(
            lower_key.as_str(),
            "token"
                | "access_token"
                | "secret"
                | "password"
                | "credential"
                | "authorization"
                | "api_key"
                | "apikey"
        ) {
            return false;
        }
        if decoded_key == "callerAgentId" {
            caller_agent_id_count += 1;
            if value.is_empty() || caller_agent_id_count > 1 {
                return false;
            }
        }
    }
    true
}

fn caller_agent_id(path_and_query: &str) -> Option<String> {
    let query = path_and_query.split_once('?')?.1;
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=')?;
        (percent_decode_ascii(key).as_deref() == Some("callerAgentId"))
            .then(|| percent_decode_ascii(value))
            .flatten()
            .filter(|value| !value.is_empty())
    })
}

fn percent_decode_ascii(value: &str) -> Option<String> {
    let mut decoded = Vec::with_capacity(value.len());
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = bytes.get(index + 1).copied().and_then(hex_value)?;
            let low = bytes.get(index + 2).copied().and_then(hex_value)?;
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn attachment_identity(servers: &[NormalizedMcpServer]) -> String {
    let mut canonical = String::new();
    for server in servers {
        canonical.push_str(&server.name);
        canonical.push('\n');
        canonical.push_str(&server.endpoint.as_url());
        canonical.push('\n');
        for header in &server.headers {
            canonical.push_str(&header.name.to_ascii_lowercase());
            canonical.push('\n');
        }
        canonical.push('\n');
    }
    fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
        bytes.iter().fold(seed, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
    }
    let bytes = canonical.as_bytes();
    format!(
        "{:016x}{:016x}",
        fnv1a(bytes, 0xcbf29ce484222325),
        fnv1a(bytes, 0x84222325cbf29ce4)
    )
}

#[derive(Clone)]
pub(crate) struct McpRegistry {
    inner: Arc<McpRegistryInner>,
}

struct McpRegistryInner {
    seats: Mutex<HashMap<String, Arc<SeatMcp>>>,
    transport_ready: bool,
}

impl McpRegistry {
    pub(crate) fn new() -> Self {
        // Probe the exact bind policy used by every seat before advertising
        // ACP HTTP MCP support.  A later per-seat bind can still fail closed.
        let transport_ready = TcpListener::bind(("127.0.0.1", 0)).is_ok();
        Self {
            inner: Arc::new(McpRegistryInner {
                seats: Mutex::new(HashMap::new()),
                transport_ready,
            }),
        }
    }

    pub(crate) fn transport_ready(&self) -> bool {
        self.inner.transport_ready
    }

    pub(crate) fn get(&self, session_id: &str) -> Option<Arc<SeatMcp>> {
        self.inner
            .seats
            .lock()
            .ok()
            .and_then(|seats| seats.get(session_id).cloned())
    }

    pub(crate) fn attach(
        &self,
        session_id: &str,
        attachment: McpAttachment,
        state_dir: &Path,
        source_gemini_dir: Option<&Path>,
        agent_profile: Option<&Path>,
        supervisor_routing_guard: bool,
    ) -> Result<Arc<SeatMcp>, ()> {
        if !self.transport_ready() || attachment.server_count() == 0 {
            return Err(());
        }
        if let Some(existing) = self.get(session_id) {
            return if existing.attachment.same_credentials(&attachment) {
                Ok(existing)
            } else {
                Err(())
            };
        }

        let root = state_dir.join("mcp-seats").join(format!(
            "{}-{}",
            sanitize_path_component(session_id),
            Uuid::new_v4()
        ));
        let home_dir = root.join("home");
        let config_path = home_dir
            .join(".gemini")
            .join("config")
            .join("mcp_config.json");
        let Some(config_parent) = config_path.parent() else {
            return Err(());
        };
        if fs::create_dir_all(config_parent).is_err() {
            let _ = fs::remove_dir_all(&root);
            return Err(());
        }
        // Keep the seat-local Gemini directory scoped for MCP, but preserve
        // the authenticated Agy runtime's non-secret configuration.  Without
        // these files the native CLI falls back to a different project/config
        // shape and can fail before producing a response.  Auth remains in
        // the OS keyring; we deliberately never copy credentials or the
        // generated mcp_config.json.
        if let Some(source) = source_gemini_dir {
            for relative in [
                Path::new("config").join("config.json"),
                Path::new("config")
                    .join("projects")
                    .join("default-cli-project.json"),
            ] {
                let source_file = source.join(&relative);
                if source_file.is_file() {
                    let destination = home_dir.join(".gemini").join(&relative);
                    let Some(parent) = destination.parent() else {
                        let _ = fs::remove_dir_all(&root);
                        return Err(());
                    };
                    if fs::create_dir_all(parent).is_err()
                        || fs::copy(&source_file, &destination).is_err()
                    {
                        let _ = fs::remove_dir_all(&root);
                        return Err(());
                    }
                }
            }
        }
        if let Some(source) = agent_profile {
            let Some(file_name) = source.file_name() else {
                let _ = fs::remove_dir_all(&root);
                return Err(());
            };
            let profile_dir = home_dir.join(".gemini").join("config").join("agents");
            if fs::create_dir_all(&profile_dir).is_err()
                || fs::copy(source, profile_dir.join(file_name)).is_err()
            {
                let _ = fs::remove_dir_all(&root);
                return Err(());
            }
        }

        let mut proxies = Vec::with_capacity(attachment.server_count());
        let mut local_urls = Vec::with_capacity(attachment.server_count());
        for server in &attachment.servers {
            // Paseo's agent-control endpoint carries the authenticated caller
            // identity that becomes the child's parent. Preserve it as the
            // authority source for Supervisor routing checks.
            let supervisor_routing_guard = supervisor_routing_guard
                && is_paseo_agent_control_endpoint(&server.endpoint.path_and_query);
            let proxy = match LoopbackProxy::bind(
                &server.endpoint,
                &server.headers,
                supervisor_routing_guard,
            ) {
                Ok(proxy) => proxy,
                Err(_) => {
                    let _ = fs::remove_dir_all(&root);
                    return Err(());
                }
            };
            local_urls.push(proxy.local_url());
            proxies.push(proxy);
        }

        let config = attachment.config_json(&local_urls);
        if write_private_json(&config_path, &config).is_err() {
            let _ = fs::remove_dir_all(&root);
            return Err(());
        }

        let seat = Arc::new(SeatMcp {
            attachment,
            root,
            home_dir,
            #[cfg(test)]
            config_path,
            proxies,
        });
        let mut seats = self.inner.seats.lock().map_err(|_| ())?;
        if let Some(existing) = seats.get(session_id) {
            if existing.attachment.same_credentials(&seat.attachment) {
                return Ok(existing.clone());
            }
            return Err(());
        }
        seats.insert(session_id.to_string(), Arc::clone(&seat));
        Ok(seat)
    }

    pub(crate) fn remove(&self, session_id: &str) {
        if let Ok(mut seats) = self.inner.seats.lock() {
            seats.remove(session_id);
        }
    }
}

pub(crate) struct SeatMcp {
    attachment: McpAttachment,
    root: PathBuf,
    home_dir: PathBuf,
    #[cfg(test)]
    config_path: PathBuf,
    proxies: Vec<LoopbackProxy>,
}

impl SeatMcp {
    pub(crate) fn identity(&self) -> &str {
        &self.attachment.identity
    }

    #[cfg(test)]
    pub(crate) fn env_vars(&self) -> Vec<(String, String)> {
        // HOME/USERPROFILE also select Agy's authentication and conversation
        // store.  Redirecting either one makes `--conversation` look in the
        // empty seat directory and can stall before the first stream event.
        Vec::new()
    }

    pub(crate) fn agy_args(&self, app_data_dir: &Path) -> Result<Vec<String>, ()> {
        let gemini_dir = self.home_dir.join(".gemini");
        if !gemini_dir.is_absolute() || !app_data_dir.is_absolute() {
            return Err(());
        }
        let relative_app_data = relative_path(&gemini_dir, app_data_dir).ok_or(())?;
        Ok(vec![
            format!("--gemini_dir={}", gemini_dir.to_string_lossy()),
            format!("--app_data_dir={}", relative_app_data.to_string_lossy()),
        ])
    }

    #[cfg(test)]
    pub(crate) fn config_path(&self) -> &Path {
        &self.config_path
    }

    #[cfg(test)]
    pub(crate) fn local_urls(&self) -> Vec<String> {
        self.proxies.iter().map(LoopbackProxy::local_url).collect()
    }
}

fn relative_path(from: &Path, to: &Path) -> Option<PathBuf> {
    let from_components: Vec<Component<'_>> = from.components().collect();
    let to_components: Vec<Component<'_>> = to.components().collect();
    let common = from_components
        .iter()
        .zip(&to_components)
        .take_while(|(left, right)| component_eq(left, right))
        .count();

    // Absolute paths on different roots/volumes cannot be represented by the
    // relative-only --app_data_dir flag.
    if common == 0
        || from_components[common..]
            .iter()
            .any(|component| matches!(component, Component::Prefix(_) | Component::RootDir))
        || to_components[common..]
            .iter()
            .any(|component| matches!(component, Component::Prefix(_) | Component::RootDir))
    {
        return None;
    }

    let mut relative = PathBuf::new();
    for component in &from_components[common..] {
        if matches!(component, Component::Normal(_)) {
            relative.push("..");
        }
    }
    for component in &to_components[common..] {
        relative.push(component.as_os_str());
    }
    Some(relative)
}

fn component_eq(left: &Component<'_>, right: &Component<'_>) -> bool {
    if cfg!(windows) {
        left.as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy())
    } else {
        left == right
    }
}

impl Drop for SeatMcp {
    fn drop(&mut self) {
        // Proxies own no credential-bearing files.  Stop them before removing
        // the seat-local config tree so a later seat cannot inherit state.
        self.proxies.clear();
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn sanitize_path_component(value: &str) -> String {
    let mut result: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .take(80)
        .collect();
    if result.is_empty() {
        result.push_str("session");
    }
    result
}

fn write_private_json(path: &Path, value: &Value) -> Result<(), io::Error> {
    let temp_path = path.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    fs::write(&temp_path, bytes)?;
    fs::rename(temp_path, path)
}

struct LoopbackProxy {
    local_addr: std::net::SocketAddr,
    local_path: String,
    stop: Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
}

impl LoopbackProxy {
    fn bind(
        endpoint: &HttpEndpoint,
        headers: &[HttpHeader],
        supervisor_routing_guard: bool,
    ) -> Result<Self, ()> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|_| ())?;
        listener.set_nonblocking(true).map_err(|_| ())?;
        let local_addr = listener.local_addr().map_err(|_| ())?;
        let local_path = format!("/mcp/{}", Uuid::new_v4());
        let stop = Arc::new(AtomicBool::new(false));
        let active_connections = Arc::new(AtomicUsize::new(0));
        let supervisor_caller_agent_id = if supervisor_routing_guard {
            Some(caller_agent_id(&endpoint.path_and_query).ok_or(())?)
        } else {
            None
        };
        let config = Arc::new(ProxyConfig {
            endpoint: endpoint.clone(),
            headers: headers.to_vec(),
            local_path: local_path.clone(),
            supervisor_caller_agent_id,
            stop: Arc::clone(&stop),
            active_connections: Arc::clone(&active_connections),
        });
        let accept_config = Arc::clone(&config);
        let accept_thread = thread::Builder::new()
            .name("agy-acp-mcp-proxy".to_string())
            .spawn(move || accept_connections(listener, accept_config))
            .map_err(|_| ())?;

        Ok(Self {
            local_addr,
            local_path,
            stop,
            accept_thread: Some(accept_thread),
        })
    }

    fn local_url(&self) -> String {
        format!(
            "http://127.0.0.1:{}{}",
            self.local_addr.port(),
            self.local_path
        )
    }
}

impl Drop for LoopbackProxy {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.local_addr);
        if let Some(thread) = self.accept_thread.take() {
            let _ = thread.join();
        }
    }
}

struct ProxyConfig {
    endpoint: HttpEndpoint,
    headers: Vec<HttpHeader>,
    local_path: String,
    supervisor_caller_agent_id: Option<String>,
    stop: Arc<AtomicBool>,
    active_connections: Arc<AtomicUsize>,
}

fn accept_connections(listener: TcpListener, config: Arc<ProxyConfig>) {
    while !config.stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _peer)) => {
                if config.active_connections.fetch_add(1, Ordering::SeqCst) >= MAX_PROXY_CONNECTIONS
                {
                    config.active_connections.fetch_sub(1, Ordering::SeqCst);
                    let _ = write_http_error(stream, 503);
                    continue;
                }
                let connection_config = Arc::clone(&config);
                let _ = thread::Builder::new()
                    .name("agy-acp-mcp-proxy-conn".to_string())
                    .spawn(move || {
                        handle_connection(stream, &connection_config);
                        connection_config
                            .active_connections
                            .fetch_sub(1, Ordering::SeqCst);
                    });
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(PROXY_ACCEPT_BACKOFF);
            }
            Err(_) => break,
        }
    }
}

struct HttpRequest {
    method: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn supervisor_policy_response(
    request: &HttpRequest,
    authenticated_caller_agent_id: &str,
) -> Option<(u16, Vec<u8>)> {
    if request.method != "POST" {
        return None;
    }
    let payload: Value = serde_json::from_slice(&request.body).ok()?;
    let batch = payload.is_array();
    let messages: Vec<&Value> = match &payload {
        Value::Array(messages) => messages.iter().collect(),
        _ => vec![&payload],
    };

    if !messages.iter().any(|message| {
        supervisor_create_agent_violation(message, authenticated_caller_agent_id).is_some()
    }) {
        return None;
    }

    let mut responses = Vec::new();
    for message in messages {
        let Some(object) = message.as_object() else {
            continue;
        };
        let id = object.get("id").cloned();
        if let Some(text) =
            supervisor_create_agent_violation(message, authenticated_caller_agent_id)
        {
            // A JSON-RPC notification has no response id, but it is still
            // withheld from Paseo.
            if let Some(id) = id {
                responses.push(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": true
                    }
                }));
            }
        } else if let Some(id) = id {
            // Do not forward only part of a JSON-RPC batch. Return an explicit
            // error for otherwise-valid members so the caller can retry them
            // separately without repeating the blocked operation.
            responses.push(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32001,
                    "message": "This request was not forwarded because the same batch contains a Supervisor SLP policy violation. Retry it separately."
                }
            }));
        }
    }

    if responses.is_empty() {
        // A batch made only of notifications has no JSON-RPC response body.
        return Some((202, Vec::new()));
    }
    let response = if batch {
        Value::Array(responses)
    } else {
        responses.into_iter().next().unwrap_or_else(|| json!([]))
    };
    serde_json::to_vec(&response).ok().map(|body| (200, body))
}

fn supervisor_create_agent_violation(
    message: &Value,
    authenticated_caller_agent_id: &str,
) -> Option<String> {
    if message.get("method").and_then(Value::as_str) != Some("tools/call")
        || message
            .get("params")
            .and_then(|params| params.get("name"))
            .and_then(Value::as_str)
            != Some("create_agent")
    {
        return None;
    }

    let provider_id = message
        .get("params")
        .and_then(|params| params.get("arguments"))
        .and_then(|arguments| arguments.get("provider"))
        .and_then(Value::as_str)
        .map(|provider| provider.split_once('/').map_or(provider, |(id, _)| id));
    if matches!(provider_id, Some("agy-lead" | "codex-root")) {
        return None;
    }

    let arguments = message
        .get("params")
        .and_then(|params| params.get("arguments"));
    let initial_prompt = arguments.and_then(|arguments| arguments.get("initialPrompt"));
    let legacy_prompt = arguments.and_then(|arguments| arguments.get("prompt"));
    let prompt = match (initial_prompt, legacy_prompt) {
        (Some(Value::String(prompt)), None) | (None, Some(Value::String(prompt))) => {
            prompt.as_str()
        }
        (Some(Value::String(prompt)), Some(Value::String(alias))) if prompt == alias => {
            prompt.as_str()
        }
        _ => "",
    };
    let notify_on_finish = arguments
        .and_then(|arguments| arguments.get("notifyOnFinish"))
        .and_then(Value::as_bool)
        == Some(true);
    let mode_id = arguments
        .and_then(|arguments| arguments.get("settings"))
        .and_then(|settings| settings.get("modeId"))
        .and_then(Value::as_str);
    let requested_role = arguments
        .and_then(|arguments| arguments.get("labels"))
        .and_then(|labels| labels.get("slpRole"))
        .and_then(Value::as_str);
    let neutral_provider = matches!(provider_id, Some("codex" | "agy-runtime" | "factory-droid"));
    if neutral_provider && requested_role.is_some() {
        let coordinator = matches!(requested_role, Some("supervisor" | "lead"));
        let read_only_mode = match provider_id {
            Some("codex") => mode_id == Some("auto"),
            Some("agy-runtime") => mode_id == Some("plan"),
            Some("factory-droid") => matches!(mode_id, Some("normal" | "spec")),
            _ => false,
        };
        if !authenticated_caller_agent_id.is_empty()
            && notify_on_finish
            && (coordinator
                || (requested_role == Some("peer")
                    && read_only_mode
                    && prompt.contains("SLP_DIRECT_PEER_CONSULTATION")
                    && prompt.contains("taskId")))
        {
            return None;
        }
        return Some("SLP policy blocked this provider-neutral assignment. Set labels.slpRole to supervisor, lead, or peer; preserve notifyOnFinish=true and the authenticated parent. Direct Peer consultation requires SLP_DIRECT_PEER_CONSULTATION, taskId, and the provider's read-only mode (Codex auto, Agy plan, Droid normal/spec).".to_string());
    }

    let custom_lead = !authenticated_caller_agent_id.is_empty()
        && provider_id == Some("factory-droid")
        && notify_on_finish
        && (prompt.contains("ROLE: LEAD") || prompt.contains("<SLP_ROLE_ASSIGNMENT>"))
        && !prompt.contains("SLP_DIRECT_PEER_CONSULTATION");
    if custom_lead {
        return None;
    }

    let valid_peer_provider_and_mode = match provider_id {
        Some("codex-peer") => mode_id == Some("auto"),
        Some("factory-droid") => matches!(
            mode_id,
            None | Some("auto" | "normal" | "spec" | "auto-low" | "auto-medium" | "auto-high")
        ),
        Some("agy") => matches!(mode_id, None | Some("auto" | "plan")),
        Some("chatgpt-web-review" | "chatgpt-web-code-lite") => {
            matches!(mode_id, None | Some("auto"))
        }
        _ => false,
    };
    let read_only_peer = !authenticated_caller_agent_id.is_empty()
        && valid_peer_provider_and_mode
        && prompt.contains("SLP_DIRECT_PEER_CONSULTATION")
        && prompt.contains("taskId")
        && notify_on_finish;
    if read_only_peer {
        return None;
    }

    Some(
        "SLP policy blocked this Agy Supervisor request before it reached Paseo. Create a Lead with `agy-lead` or `codex-root` (or owner-selected `factory-droid`), or use a direct Peer consultation route (`codex-peer`, `factory-droid`, `agy`, `chatgpt-web-review`, or `chatgpt-web-code-lite`) with `SLP_DIRECT_PEER_CONSULTATION`, `taskId`, `settings.modeId=auto`, and `notifyOnFinish=true`. The MCP caller identity supplies the actual parent; do not copy or invent a parent ID in the prompt. Include `leadAgentId` only when the task already has a matching Lead. Peer edits and acceptance remain Lead-owned. Do not retry through another provider or route.".to_string(),
    )
}

fn normalize_create_agent_message(message: &mut Value) -> bool {
    if message.get("method").and_then(Value::as_str) != Some("tools/call")
        || message
            .get("params")
            .and_then(|params| params.get("name"))
            .and_then(Value::as_str)
            != Some("create_agent")
    {
        return false;
    }

    let Some(arguments) = message
        .get_mut("params")
        .and_then(Value::as_object_mut)
        .and_then(|params| params.get_mut("arguments"))
        .and_then(Value::as_object_mut)
    else {
        return false;
    };

    let provider_id = arguments
        .get("provider")
        .and_then(Value::as_str)
        .map(|provider| {
            provider
                .split_once('/')
                .map_or(provider, |(id, _)| id)
                .to_string()
        });
    let current_mode = arguments
        .get("settings")
        .and_then(Value::as_object)
        .and_then(|settings| settings.get("modeId"))
        .and_then(Value::as_str)
        .map(str::to_string);

    let normalized_mode = match (provider_id.as_deref(), current_mode.as_deref()) {
        (Some("factory-droid"), None | Some("auto" | "full-access")) => Some("auto-high"),
        (Some("agy"), None | Some("auto")) => Some("plan"),
        (Some("agy"), Some("full-access")) => Some("accept-edits"),
        (Some("codex-peer" | "chatgpt-web-review" | "chatgpt-web-code-lite"), None) => Some("auto"),
        _ => None,
    };

    let Some(normalized_mode) = normalized_mode else {
        return false;
    };

    let settings = arguments
        .entry("settings")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !settings.is_object() {
        *settings = Value::Object(serde_json::Map::new());
    }
    if let Some(settings_obj) = settings.as_object_mut() {
        settings_obj.insert(
            "modeId".to_string(),
            Value::String(normalized_mode.to_string()),
        );
        return true;
    }
    false
}

fn normalize_outbound_request(request: &mut HttpRequest) {
    if request.method != "POST" {
        return;
    }
    let Ok(mut payload) = serde_json::from_slice::<Value>(&request.body) else {
        return;
    };
    let changed = match &mut payload {
        Value::Array(messages) => {
            let mut any = false;
            for message in messages {
                if normalize_create_agent_message(message) {
                    any = true;
                }
            }
            any
        }
        message => normalize_create_agent_message(message),
    };
    if changed {
        if let Ok(encoded) = serde_json::to_vec(&payload) {
            request.body = encoded;
        }
    }
}

fn write_http_json_response(mut stream: TcpStream, status: u16, body: &[u8]) -> io::Result<()> {
    let (reason, content_type) = if status == 202 {
        ("Accepted", "")
    } else {
        ("OK", "Content-Type: application/json\r\n")
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nConnection: close\r\n{content_type}Content-Length: {}\r\n\r\n",
        body.len(),
    )?;
    stream.write_all(body)
}

fn handle_connection(mut local: TcpStream, config: &ProxyConfig) {
    let _ = local.set_read_timeout(Some(PROXY_READ_TIMEOUT));
    let _ = local.set_write_timeout(Some(PROXY_READ_TIMEOUT));
    let mut request = match read_request(&mut local, &config.local_path) {
        Ok(request) => request,
        Err(_) => {
            let _ = write_http_error(local, 400);
            return;
        }
    };

    if let Some(authenticated_caller_agent_id) = &config.supervisor_caller_agent_id {
        if let Some((status, response)) =
            supervisor_policy_response(&request, authenticated_caller_agent_id)
        {
            let _ = write_http_json_response(local, status, &response);
            return;
        }
    }
    normalize_outbound_request(&mut request);

    let connect_deadline = Instant::now() + PROXY_CONNECT_TIMEOUT;
    let mut remote = None;
    let addresses = match (config.endpoint.host.as_str(), config.endpoint.port).to_socket_addrs() {
        Ok(addresses) => addresses,
        Err(_) => {
            let _ = write_http_error(local, 502);
            return;
        }
    };
    for address in addresses {
        let remaining = connect_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        if let Ok(stream) = TcpStream::connect_timeout(&address, remaining) {
            remote = Some(stream);
            break;
        }
    }
    let Some(mut remote) = remote else {
        let _ = write_http_error(local, 502);
        return;
    };
    let _ = remote.set_read_timeout(Some(PROXY_READ_TIMEOUT));
    let _ = remote.set_write_timeout(Some(PROXY_READ_TIMEOUT));
    if write_upstream_request(&mut remote, &request, &config.endpoint, &config.headers).is_err() {
        let _ = write_http_error(local, 502);
        return;
    }
    let _ = forward_upstream_response(&mut remote, &mut local, &config.stop);
    let _ = remote.shutdown(Shutdown::Both);
    let _ = local.shutdown(Shutdown::Both);
}

fn read_request(stream: &mut TcpStream, expected_path: &str) -> Result<HttpRequest, ()> {
    let (header_bytes, mut buffered_body) = read_header_block(stream)?;
    let header_text = std::str::from_utf8(&header_bytes).map_err(|_| ())?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().ok_or(())?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().ok_or(())?.to_string();
    let target = request_parts.next().ok_or(())?.to_string();
    let version = request_parts.next().ok_or(())?;
    if request_parts.next().is_some()
        || !matches!(version, "HTTP/1.0" | "HTTP/1.1")
        || !matches!(method.as_str(), "GET" | "POST" | "DELETE" | "OPTIONS")
        || target != expected_path
    {
        return Err(());
    }

    let mut headers = Vec::new();
    let mut content_length = 0usize;
    let mut has_transfer_encoding = false;
    for line in lines {
        if line.is_empty() {
            break;
        }
        let (name, value) = line.split_once(':').ok_or(())?;
        let name = name.trim();
        let value = value.trim();
        if !valid_header_name(name) || !valid_header_value(value) {
            return Err(());
        }
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.parse::<usize>().map_err(|_| ())?;
            if content_length > MAX_REQUEST_BODY {
                return Err(());
            }
        }
        if name.eq_ignore_ascii_case("transfer-encoding") {
            has_transfer_encoding = true;
        }
        headers.push((name.to_string(), value.to_string()));
    }
    if has_transfer_encoding {
        return Err(());
    }

    let body_deadline = Instant::now() + PROXY_HEADER_TIMEOUT;
    while buffered_body.len() < content_length {
        let remaining = content_length - buffered_body.len();
        let mut chunk = vec![0u8; remaining.min(16 * 1024)];
        match stream.read(&mut chunk) {
            Ok(0) => return Err(()),
            Ok(read) => buffered_body.extend_from_slice(&chunk[..read]),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                if Instant::now() >= body_deadline {
                    return Err(());
                }
            }
            Err(_) => return Err(()),
        }
    }
    buffered_body.truncate(content_length);
    Ok(HttpRequest {
        method,
        headers,
        body: buffered_body,
    })
}

fn read_header_block(stream: &mut TcpStream) -> Result<(Vec<u8>, Vec<u8>), ()> {
    let mut bytes = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let deadline = Instant::now() + PROXY_HEADER_TIMEOUT;
    let end = loop {
        if bytes.len() > MAX_HTTP_HEADERS {
            return Err(());
        }
        match stream.read(&mut chunk) {
            Ok(0) => return Err(()),
            Ok(read) => {
                bytes.extend_from_slice(&chunk[..read]);
                if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    break position;
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                if Instant::now() >= deadline {
                    return Err(());
                }
                continue;
            }
            Err(_) => return Err(()),
        }
    };
    let body_start = end + 4;
    if body_start > MAX_HTTP_HEADERS {
        return Err(());
    }
    Ok((bytes[..body_start].to_vec(), bytes[body_start..].to_vec()))
}

fn write_upstream_request(
    remote: &mut TcpStream,
    request: &HttpRequest,
    endpoint: &HttpEndpoint,
    configured_headers: &[HttpHeader],
) -> Result<(), ()> {
    let mut output = Vec::with_capacity(4096 + request.body.len());
    output.extend_from_slice(request.method.as_bytes());
    output.push(b' ');
    output.extend_from_slice(endpoint.path_and_query.as_bytes());
    output.extend_from_slice(b" HTTP/1.1\r\nHost: ");
    output.extend_from_slice(endpoint.authority.as_bytes());
    output.extend_from_slice(b"\r\nConnection: close\r\n");

    let configured_names: HashSet<String> = configured_headers
        .iter()
        .map(|header| header.name.to_ascii_lowercase())
        .collect();
    for (name, value) in &request.headers {
        let lower_name = name.to_ascii_lowercase();
        if matches!(
            lower_name.as_str(),
            "host"
                | "connection"
                | "keep-alive"
                | "proxy-connection"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailer"
                | "transfer-encoding"
                | "upgrade"
                | "content-length"
                | "authorization"
        ) || configured_names.contains(&lower_name)
        {
            continue;
        }
        output.extend_from_slice(name.as_bytes());
        output.extend_from_slice(b": ");
        output.extend_from_slice(value.as_bytes());
        output.extend_from_slice(b"\r\n");
    }
    for header in configured_headers {
        output.extend_from_slice(header.name.as_bytes());
        output.extend_from_slice(b": ");
        output.extend_from_slice(header.value.as_bytes());
        output.extend_from_slice(b"\r\n");
    }
    output.extend_from_slice(b"Content-Length: ");
    output.extend_from_slice(request.body.len().to_string().as_bytes());
    output.extend_from_slice(b"\r\n\r\n");
    output.extend_from_slice(&request.body);
    remote.write_all(&output).map_err(|_| ())
}

fn forward_upstream_response(
    remote: &mut TcpStream,
    local: &mut TcpStream,
    stop: &AtomicBool,
) -> Result<(), ()> {
    let (header_bytes, mut buffered_body) = read_header_block(remote)?;
    local.write_all(&header_bytes).map_err(|_| ())?;

    let header_text = std::str::from_utf8(&header_bytes).map_err(|_| ())?;
    let mut content_length = None;
    for line in header_text.split("\r\n").skip(1) {
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(());
        };
        if name.eq_ignore_ascii_case("content-length") {
            content_length = Some(value.trim().parse::<usize>().map_err(|_| ())?);
        }
    }

    if let Some(content_length) = content_length {
        forward_exact_body(remote, local, &mut buffered_body, content_length, stop)?;
    } else {
        // A streamable HTTP/SSE response commonly has no content length.  Agy
        // receives the bytes as they arrive; the upstream close terminates it.
        forward_until_close(remote, local, &mut buffered_body, stop)?;
    }
    Ok(())
}

fn forward_exact_body(
    remote: &mut TcpStream,
    local: &mut TcpStream,
    buffered: &mut Vec<u8>,
    length: usize,
    stop: &AtomicBool,
) -> Result<(), ()> {
    let mut written = 0usize;
    if !buffered.is_empty() {
        let take = buffered.len().min(length);
        local.write_all(&buffered[..take]).map_err(|_| ())?;
        written += take;
        buffered.drain(..take);
    }
    let mut chunk = [0u8; 16 * 1024];
    while written < length && !stop.load(Ordering::SeqCst) {
        match remote.read(&mut chunk) {
            Ok(0) => return Err(()),
            Ok(read) => {
                let remaining = length - written;
                let take = read.min(remaining);
                local.write_all(&chunk[..take]).map_err(|_| ())?;
                written += take;
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue
            }
            Err(_) => return Err(()),
        }
    }
    if written == length {
        Ok(())
    } else {
        Err(())
    }
}

fn forward_until_close(
    remote: &mut TcpStream,
    local: &mut TcpStream,
    buffered: &mut Vec<u8>,
    stop: &AtomicBool,
) -> Result<(), ()> {
    if !buffered.is_empty() {
        local.write_all(buffered).map_err(|_| ())?;
        buffered.clear();
    }
    let mut chunk = [0u8; 16 * 1024];
    while !stop.load(Ordering::SeqCst) {
        match remote.read(&mut chunk) {
            Ok(0) => return Ok(()),
            Ok(read) => local.write_all(&chunk[..read]).map_err(|_| ())?,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue
            }
            Err(_) => return Err(()),
        }
    }
    Ok(())
}

fn write_http_error(mut stream: TcpStream, status: u16) -> io::Result<()> {
    let reason = match status {
        400 => "Bad Request",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    };
    let body = b"MCP proxy unavailable";
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nConnection: close\r\nContent-Length: {}\r\nContent-Type: text/plain\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_provider_roles_and_consultation_modes() {
        for provider in ["codex", "agy-runtime", "factory-droid"] {
            for role in ["supervisor", "lead"] {
                let message = json!({"method":"tools/call","params":{"name":"create_agent","arguments":{
                    "provider":provider,"labels":{"slpRole":role},"notifyOnFinish":true,
                    "initialPrompt":"bounded task"
                }}});
                assert!(supervisor_create_agent_violation(&message, "authenticated").is_none());
                assert!(supervisor_create_agent_violation(&message, "").is_some());
            }
            let mode = match provider {
                "codex" => "auto",
                "agy-runtime" => "plan",
                _ => "normal",
            };
            let mut message = json!({"method":"tools/call","params":{"name":"create_agent","arguments":{
                "provider":provider,"labels":{"slpRole":"peer"},"notifyOnFinish":true,
                "initialPrompt":"SLP_DIRECT_PEER_CONSULTATION taskId=bounded",
                "settings":{"modeId":mode}
            }}});
            assert!(supervisor_create_agent_violation(&message, "authenticated").is_none());
            message["params"]["arguments"]["settings"]["modeId"] = json!("full-access");
            assert!(supervisor_create_agent_violation(&message, "authenticated").is_some());
            message["params"]["arguments"]["labels"]["slpRole"] = json!("admin");
            assert!(supervisor_create_agent_violation(&message, "authenticated").is_some());
        }
    }

    #[test]
    fn test_mcp_receipt_consultation_uses_paseo_initial_prompt_without_existing_lead() {
        let request = HttpRequest {
            method: "POST".into(),
            headers: Vec::new(),
            body: serde_json::to_vec(&json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "create_agent", "arguments": {
                    "provider": "codex-peer/gpt-6-luna",
                    "initialPrompt": "SLP_DIRECT_PEER_CONSULTATION taskId=consult-1",
                    "settings": {"modeId": "auto"}, "notifyOnFinish": true
                }}
            }))
            .unwrap(),
        };
        assert!(supervisor_policy_response(&request, "authenticated-supervisor").is_none());
        assert!(supervisor_policy_response(&request, "").is_some());
    }

    #[test]
    fn test_mcp_receipt_consultation_rejects_conflicting_prompt_aliases() {
        let request = HttpRequest {
            method: "POST".into(),
            headers: Vec::new(),
            body: serde_json::to_vec(&json!({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "create_agent", "arguments": {
                    "provider": "codex-peer",
                    "initialPrompt": "Write project files",
                    "prompt": "SLP_DIRECT_PEER_CONSULTATION taskId=consult-1",
                    "settings": {"modeId": "auto"}, "notifyOnFinish": true
                }}
            }))
            .unwrap(),
        };
        assert!(supervisor_policy_response(&request, "authenticated-supervisor").is_some());
    }

    #[test]
    fn test_mcp_supervisor_allows_factory_droid_peer_and_normalizes_auto_mode() {
        let mut request = HttpRequest {
            method: "POST".into(),
            headers: Vec::new(),
            body: serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "create_agent",
                    "arguments": {
                        "provider": "factory-droid/gpt-6.1-sol",
                        "initialPrompt": "SLP_DIRECT_PEER_CONSULTATION\ntaskId: CAMPAIGN-START-QUOTA-STREAMING-AUDIO\nleadAgentId: f23bc5d0-b39b-4d8f-968f-cbb91fcbbdd1",
                        "settings": {"modeId": "auto", "thinkingOptionId": "high"},
                        "notifyOnFinish": true
                    }
                }
            }))
            .unwrap(),
        };
        assert!(supervisor_policy_response(&request, "authenticated-supervisor").is_none());
        normalize_outbound_request(&mut request);
        let forwarded: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(
            forwarded["params"]["arguments"]["settings"]["modeId"],
            "auto-high"
        );
        assert_eq!(
            forwarded["params"]["arguments"]["settings"]["thinkingOptionId"],
            "high"
        );
    }
}
