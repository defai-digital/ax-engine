use std::time::Duration;

/// Default connect timeout for delegated local/remote HTTP backends.
pub const DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS: u64 = 30;
/// Default read/write timeout for delegated HTTP I/O. Reads intentionally share
/// the longer I/O timeout because streaming completions can stay open for the
/// full generation window.
pub const DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS: u64 = 300;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DelegatedHttpTimeouts {
    pub connect: Duration,
    pub read: Duration,
    pub write: Duration,
}

impl DelegatedHttpTimeouts {
    pub fn from_secs(connect_secs: u64, read_secs: u64, write_secs: u64) -> Self {
        Self {
            connect: Duration::from_secs(connect_secs),
            read: Duration::from_secs(read_secs),
            write: Duration::from_secs(write_secs),
        }
    }

    pub fn default_connect_secs() -> u64 {
        DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS
    }

    pub fn default_io_secs() -> u64 {
        DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS
    }

    pub(crate) fn build_agent(self) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout_connect(self.connect)
            .timeout_read(self.read)
            .timeout_write(self.write)
            .build()
    }
}

impl Default for DelegatedHttpTimeouts {
    fn default() -> Self {
        Self::from_secs(
            DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS,
            DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS,
            DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS,
        )
    }
}
