//! Opt-in mDNS / DNS-SD advertisement for LAN discovery by AX Serving.
//!
//! Service type: `_ax-engine._tcp`
//! See `docs/LAN-DISCOVERY.md`.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};

use mdns_sd::{ServiceDaemon, ServiceInfo};
use tracing::{info, warn};

/// DNS-SD service type without the trailing `.local.` (mdns-sd appends domain).
pub const ENGINE_SERVICE_TYPE: &str = "_ax-engine._tcp.local.";
pub const DISCOVERY_PROTO: &str = "1";
pub const DISCOVERY_KIND: &str = "ax_engine";

#[derive(Debug, Clone)]
pub struct LanAdvertiseConfig {
    pub instance_name: String,
    pub port: u16,
    pub advertise_ip: Ipv4Addr,
    pub version: String,
    pub model_id: String,
    pub auth_required: bool,
    pub cluster: Option<String>,
    pub instance_id: String,
}

pub struct LanAdvertiser {
    daemon: ServiceDaemon,
    fullname: String,
    config: parking_lot::Mutex<LanAdvertiseConfig>,
}

pub(crate) trait ModelAdvertisement: Send + Sync {
    fn update_model(&self, model_id: &str) -> Result<(), String>;
}

impl LanAdvertiser {
    pub fn start(config: LanAdvertiseConfig) -> Result<Self, String> {
        let daemon = ServiceDaemon::new().map_err(|err| format!("mdns daemon: {err}"))?;
        let service = service_info(&config)?;

        let fullname = service.get_fullname().to_string();
        daemon
            .register(service)
            .map_err(|err| format!("mdns register: {err}"))?;

        info!(
            service = %fullname,
            advertise_ip = %config.advertise_ip,
            port = config.port,
            "ax-engine LAN mDNS advertisement registered"
        );

        Ok(Self {
            daemon,
            fullname,
            config: parking_lot::Mutex::new(config),
        })
    }
}

impl ModelAdvertisement for LanAdvertiser {
    fn update_model(&self, model_id: &str) -> Result<(), String> {
        let mut config = self.config.lock();
        if config.model_id == model_id {
            return Ok(());
        }
        let mut updated = config.clone();
        updated.model_id = model_id.to_string();
        self.daemon
            .register(service_info(&updated)?)
            .map_err(|err| format!("mdns re-register: {err}"))?;
        *config = updated;
        Ok(())
    }
}

fn service_info(config: &LanAdvertiseConfig) -> Result<ServiceInfo, String> {
    let mut properties: HashMap<String, String> = HashMap::new();
    properties.insert("proto".into(), DISCOVERY_PROTO.into());
    properties.insert("kind".into(), DISCOVERY_KIND.into());
    properties.insert("version".into(), config.version.clone());
    properties.insert("model".into(), config.model_id.clone());
    properties.insert(
        "auth".into(),
        if config.auth_required {
            "required".into()
        } else {
            "open".into()
        },
    );
    properties.insert("scheme".into(), "http".into());
    properties.insert("path".into(), "/v1".into());
    properties.insert("instance".into(), config.instance_id.clone());
    properties.insert("platform".into(), current_platform());
    if let Some(cluster) = config
        .cluster
        .as_ref()
        .filter(|cluster| !cluster.is_empty())
    {
        properties.insert("cluster".into(), cluster.clone());
    }

    let instance_name = sanitize_instance_name(&config.instance_name);
    let host_name = format!("{instance_name}.local.");
    ServiceInfo::new(
        ENGINE_SERVICE_TYPE,
        &instance_name,
        &host_name,
        IpAddr::V4(config.advertise_ip),
        config.port,
        Some(properties),
    )
    .map_err(|err| format!("mdns service info: {err}"))
}

impl Drop for LanAdvertiser {
    fn drop(&mut self) {
        if let Err(err) = self.daemon.unregister(&self.fullname) {
            warn!(error = %err, fullname = %self.fullname, "failed to unregister mDNS service");
        }
        // Best-effort shutdown; ignore errors during process exit.
        let _ = self.daemon.shutdown();
    }
}

pub fn sanitize_instance_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() || ch == '.' {
            out.push('-');
        }
    }
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "ax-engine".into()
    } else {
        out.chars().take(63).collect()
    }
}

pub fn current_platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Prefer a stable private IPv4 for advertisement. Skips loopback and link-local.
pub fn pick_advertise_ipv4(explicit: Option<&str>, bind_host: &str) -> Result<Ipv4Addr, String> {
    if let Some(raw) = explicit {
        let ip: IpAddr = raw
            .parse()
            .map_err(|err| format!("invalid --lan-advertise-host {raw}: {err}"))?;
        match ip {
            IpAddr::V4(v4) if is_advertisable_v4(v4) => return Ok(v4),
            IpAddr::V4(v4) => {
                return Err(format!(
                    "lan advertise host {v4} is not a private unicast IPv4 suitable for fleet join"
                ));
            }
            IpAddr::V6(_) => {
                return Err(
                    "lan advertise host must be IPv4 in phase 1 (set a private IPv4)".into(),
                );
            }
        }
    }

    if let Ok(ip) = bind_host.parse::<Ipv4Addr>()
        && is_advertisable_v4(ip)
    {
        return Ok(ip);
    }

    // Scan interfaces via local_ip_address when available; fallback: enumerate
    // common route by connecting a UDP socket to a public DNS (no packets sent).
    if let Some(ip) = guess_private_ipv4() {
        return Ok(ip);
    }

    Err("could not determine a private IPv4 for LAN advertise; set --lan-advertise-host".into())
}

fn is_advertisable_v4(ip: Ipv4Addr) -> bool {
    !ip.is_unspecified()
        && !ip.is_loopback()
        && !ip.is_multicast()
        && !ip.is_broadcast()
        && !ip.is_link_local()
        && (ip.is_private() || is_cgnat_v4(ip))
}

/// CGNAT shared address space 100.64.0.0/10 (stable on current Rust toolchains).
fn is_cgnat_v4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 100 && (octets[1] & 0xc0) == 64
}

fn guess_private_ipv4() -> Option<Ipv4Addr> {
    // UDP connect does not send traffic; it selects a source address via routing.
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("1.1.1.1:80").ok()?;
    match socket.local_addr().ok()?.ip() {
        IpAddr::V4(v4) if is_advertisable_v4(v4) => Some(v4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_collapses_junk() {
        assert_eq!(sanitize_instance_name("Mac Studio #2"), "Mac-Studio-2");
        assert_eq!(sanitize_instance_name("@@@"), "ax-engine");
    }

    #[test]
    fn private_ipv4_accepted() {
        assert!(is_advertisable_v4(Ipv4Addr::new(192, 168, 1, 10)));
        assert!(!is_advertisable_v4(Ipv4Addr::LOCALHOST));
        assert!(!is_advertisable_v4(Ipv4Addr::new(169, 254, 1, 1)));
    }

    #[test]
    fn explicit_advertise_host_parses() {
        let ip = pick_advertise_ipv4(Some("10.0.0.5"), "127.0.0.1").unwrap();
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 5));
    }

    #[test]
    fn cgnat_and_reject_public_ipv4() {
        assert!(is_advertisable_v4(Ipv4Addr::new(100, 64, 1, 1)));
        assert!(!is_advertisable_v4(Ipv4Addr::new(8, 8, 8, 8)));
        assert!(!is_advertisable_v4(Ipv4Addr::UNSPECIFIED));
    }

    #[test]
    fn service_info_uses_current_default_model() {
        let mut config = LanAdvertiseConfig {
            instance_name: "test-engine".to_string(),
            port: 8080,
            advertise_ip: Ipv4Addr::new(192, 168, 1, 10),
            version: "6.9.0".to_string(),
            model_id: "first".to_string(),
            auth_required: true,
            cluster: Some("test".to_string()),
            instance_id: "instance".to_string(),
        };
        assert_eq!(
            service_info(&config)
                .expect("service info should build")
                .get_property_val_str("model"),
            Some("first")
        );

        config.model_id = "second".to_string();
        assert_eq!(
            service_info(&config)
                .expect("updated service info should build")
                .get_property_val_str("model"),
            Some("second")
        );
    }
}
