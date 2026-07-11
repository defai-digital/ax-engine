//! One module per screen; each extends `App` with its `on_key_*`/`draw_*`
//! handlers so all state stays on the single `App` struct in `tui::mod`.

pub(super) mod chat;
pub(super) mod downloads;
pub(super) mod home;
pub(super) mod metrics_panel;
pub(super) mod models;
pub(super) mod serve;
