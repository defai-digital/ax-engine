#!/usr/bin/env bash
# Source this before Cargo or maturin so child tools use the same Rust pin.
ax_use_pinned_rust() {
    local repo_root="$1"
    local pin rustc_path tool_dir tool actual
    pin="$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([0-9]+\.[0-9]+\.[0-9]+)"[[:space:]]*$/\1/p' "$repo_root/rust-toolchain.toml")" || return 1
    if [[ ! "$pin" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "error: missing or invalid Rust pin in $repo_root/rust-toolchain.toml" >&2
        return 1
    fi
    rustc_path="$(rustup which --toolchain "$pin" rustc)" || return 1
    tool_dir="$(dirname "$rustc_path")"
    for tool in rustc cargo rustdoc; do
        actual="$("$tool_dir/$tool" --version)" || return 1
        if [[ "$actual" != "$tool $pin "* ]]; then
            echo "error: expected $tool $pin, found $actual" >&2
            return 1
        fi
    done
    for tool in cargo-clippy cargo-fmt; do
        if [[ ! -x "$tool_dir/$tool" ]]; then
            echo "error: pinned $tool is missing; install the components in rust-toolchain.toml" >&2
            return 1
        fi
    done
    export PATH="$tool_dir:$PATH"
    export RUSTC="$tool_dir/rustc"
    export RUSTDOC="$tool_dir/rustdoc"
    export CARGO="$tool_dir/cargo"
    export RUSTUP_TOOLCHAIN="$pin"
    AX_CARGO_BIN="$CARGO"
}
