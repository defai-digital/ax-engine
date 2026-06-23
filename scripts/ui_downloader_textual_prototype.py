#!/usr/bin/env python3
"""Prototype: a Textual TUI version of the `ax-engine ui-downloader` picker.

This is an evaluation prototype, NOT shipping code. It reuses the real model
catalog from `ax_engine._cli` so the look/feel can be compared against the
line-based wizard without duplicating data. It does not download anything; on
selection it prints the command it *would* run and exits.

Run it (Textual must be installed in the environment):

    python scripts/ui_downloader_textual_prototype.py

Keys: arrows move, Enter selects, q quits. For an MTP-capable model a modal
offers Direct vs MTP.
"""
from __future__ import annotations

import sys

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, DataTable, Footer, Header, Label
except ImportError:  # pragma: no cover - prototype-only dependency
    sys.exit(
        "Textual is not installed. Install it to try this prototype:\n"
        "  pip install textual"
    )

# Reuse the production catalog so the prototype stays in sync with the real CLI.
from ax_engine._cli import _default_download_root, _downloadable_profiles


class VariantScreen(ModalScreen[str | None]):
    """Direct-vs-MTP choice for a model that has an MTP acceleration package."""

    CSS = """
    VariantScreen { align: center middle; }
    #dialog {
        width: 78; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #dialog Label { margin-bottom: 1; }
    #dialog Button { width: 100%; margin-bottom: 1; }
    """

    def __init__(self, profile) -> None:
        super().__init__()
        self.profile = profile

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"[b]{self.profile.label}[/b] has an MTP acceleration package.")
            yield Button(
                f"Direct download   ({self.profile.repo_id})",
                id="direct",
                variant="primary",
            )
            yield Button(
                f"MTP package   (download-mtp {self.profile.mtp_target})",
                id="mtp",
                variant="success",
            )
            yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None if event.button.id == "cancel" else event.button.id)


class DownloaderApp(App):
    """Minimal model picker. Exits with (label, variant, target) or None."""

    TITLE = "AX Engine — download a model"
    CSS = "DataTable { height: 1fr; }"
    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.profiles = _downloadable_profiles()

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="models", zebra_stripes=True)
        yield Label(f"Default destination: {_default_download_root()}", id="dest")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "Model", "MTP", "Repo")
        for index, profile in enumerate(self.profiles, start=1):
            table.add_row(
                str(index),
                profile.label,
                "yes" if profile.mtp_target else "—",
                profile.repo_id,
                key=str(index - 1),
            )
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        profile = self.profiles[int(event.row_key.value)]
        if profile.mtp_target:
            def resolve(choice: str | None) -> None:
                if choice is not None:
                    target = profile.mtp_target if choice == "mtp" else profile.repo_id
                    self.exit((profile.label, choice, target))

            self.push_screen(VariantScreen(profile), resolve)
        else:
            self.exit((profile.label, "direct", profile.repo_id))


def main() -> int:
    result = DownloaderApp().run()
    if result is None:
        print("Cancelled.")
        return 130
    label, variant, target = result
    if variant == "mtp":
        print(f"Would run: ax-engine download-mtp {target}")
    else:
        print(f"Would run: ax-engine download {label}   (repo: {target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
