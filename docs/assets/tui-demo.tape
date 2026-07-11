# AX Engine TUI demo capture (VHS).
# Render from repo root:
#   bash scripts/capture-tui.sh
# or:
#   cargo build -p ax-engine-bench --bin ax-engine && vhs docs/assets/tui-demo.tape
#
# Outputs (relative to repo root when vhs is run there):
#   docs/assets/tui-demo.gif  — animated README clip
# scripts/capture-tui.sh also writes docs/assets/tui-demo.png (still from the GIF).

Output docs/assets/tui-demo.gif

Set Shell "bash"
Set FontSize 13
Set Width 1440
Set Height 900
Set Padding 14
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set Framerate 12
Set PlaybackSpeed 1.0
Set WindowBar "Rings"
Set WindowBarSize 40
Set BorderRadius 8

# Binary is built by scripts/capture-tui.sh (./target/debug/ax-engine).
Require env

Hide
Type "export TERM=xterm-256color COLORTERM=truecolor"
Enter
Type "clear"
Enter
Show

Type "./target/debug/ax-engine tui"
Enter

# First paint + identity strip.
Sleep 2s
# Host metrics sample every ~2s; wait so the utilization chart has a trail.
Sleep 10s

# Models tab briefly, then back to Home (chart + top-10 memory).
Type "2"
Sleep 2s
Type "1"
Sleep 4s

# Clean quit (no jobs → no confirm modal).
Type "q"
Sleep 500ms
