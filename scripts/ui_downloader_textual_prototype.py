#!/usr/bin/env python3
"""Prototype: a Textual TUI for `ax-engine ui-downloader` that actually downloads.

Evaluation prototype, NOT shipping code. It reuses the real model catalog from
`ax_engine._cli` and the real download logic from `scripts/download_model.py`,
so the look/feel and the worker-thread integration can be judged against the
line-based wizard.

Flow: pick a model (DataTable) -> for an MTP-capable model choose Direct vs MTP
-> a download screen runs the real download:
  * Direct: in-process Hugging Face download on a worker thread, with a live
    Textual ProgressBar plus a bytes/speed line.
  * MTP: runs `ax-engine download-mtp <target>` and streams its output into a log.

Run it (Textual must be installed):

    PYTHONPATH=scripts python scripts/ui_downloader_textual_prototype.py

Keys: arrows move, Enter selects, q quits, b goes back.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.screen import ModalScreen, Screen
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Label,
        ProgressBar,
        RichLog,
    )
    from textual.worker import get_current_worker
except ImportError:  # pragma: no cover - prototype-only dependency
    sys.exit(
        "Textual is not installed. Install it to try this prototype:\n"
        "  pip install textual"
    )

# Reuse the production catalog so the prototype stays in sync with the real CLI.
from ax_engine._cli import _default_download_root, _downloadable_profiles


def _load_download_model():
    """Import scripts/download_model.py by path (it is a script, not a package)."""
    path = Path(__file__).resolve().parent / "download_model.py"
    spec = importlib.util.spec_from_file_location("ax_dm_proto", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Swappable so headless tests can inject a fake download backend.
_DM = _load_download_model()


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


class DownloadScreen(Screen):
    """Runs the real download for one model and reports progress live."""

    BINDINGS = [Binding("q", "quit", "Quit"), Binding("b", "back", "Back")]

    def __init__(self, profile, variant: str) -> None:
        super().__init__()
        self.profile = profile
        self.variant = variant
        self._total: int | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("", id="status")
        if self.variant == "direct":
            yield ProgressBar(total=None, show_eta=True, id="bar")
            yield Label("", id="detail")
        else:
            yield RichLog(id="log", markup=False, highlight=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#status", Label).update(f"Downloading {self.profile.repo_id}")
        if self.variant == "direct":
            self.run_direct()
        else:
            self.run_mtp()

    def action_back(self) -> None:
        self.dismiss()

    # ----- direct download (in-process, worker thread + disk-poll progress) -----

    @work(thread=True, exclusive=True)
    def run_direct(self) -> None:
        worker = get_current_worker()
        dm = _DM
        repo = self.profile.repo_id
        repo_dir = dm.default_mlx_lm_repo_cache_dir(repo)
        try:
            total = dm._total_repo_bytes(repo)
        except Exception:
            total = None
        self.app.call_from_thread(self._set_total, total)

        err: dict[str, BaseException] = {}

        def _do() -> None:
            try:
                dm.download(repo, None, quiet=True)
            except BaseException as exc:  # noqa: BLE001 - surface any failure to the UI
                err["e"] = exc

        downloader = threading.Thread(target=_do, daemon=True)
        downloader.start()

        last_t: float | None = None
        last_b = 0
        ema: float | None = None
        while downloader.is_alive():
            if worker.is_cancelled:
                break
            now = time.monotonic()
            downloaded = dm._dir_size_bytes(repo_dir)
            speed: float | None = None
            if last_t is not None:
                dt = now - last_t
                if dt > 0:
                    inst = max(downloaded - last_b, 0) / dt
                    ema = inst if ema is None else 0.6 * ema + 0.4 * inst
                    speed = ema
            last_t, last_b = now, downloaded
            self.app.call_from_thread(self._update, downloaded, speed)
            time.sleep(0.4)

        downloader.join(timeout=1.0)
        self.app.call_from_thread(self._finish, err.get("e"), dm._dir_size_bytes(repo_dir))

    def _set_total(self, total: int | None) -> None:
        self._total = total
        self.query_one(ProgressBar).update(total=total)

    def _update(self, downloaded: int, speed: float | None) -> None:
        if self._total:
            self.query_one(ProgressBar).update(progress=min(downloaded, self._total))
        self.query_one("#detail", Label).update(self._detail(downloaded, speed))

    def _detail(self, downloaded: int, speed: float | None) -> str:
        dm = _DM
        size = (
            f"{dm._format_bytes(downloaded)} / {dm._format_bytes(self._total)}"
            if self._total
            else dm._format_bytes(downloaded)
        )
        rate = f"{dm._format_bytes(speed)}/s" if speed else "-- B/s"
        return f"{size}   ·   {rate}"

    def _finish(self, err: BaseException | None, downloaded: int) -> None:
        status = self.query_one("#status", Label)
        if err is not None:
            status.update(f"[red]Failed:[/red] {err}")
            return
        if self._total:
            self.query_one(ProgressBar).update(progress=self._total)
        self.query_one("#detail", Label).update(self._detail(downloaded, None))
        status.update("[green]Done.[/green]   Press b for back, q to quit.")

    # ----- MTP package (shell out to the real download-mtp, stream output) -----

    @work(thread=True, exclusive=True)
    def run_mtp(self) -> None:
        from ax_engine import _cli

        log = self.query_one(RichLog)
        argv = [str(_cli._bench_bin()), "download-mtp", self.profile.mtp_target]
        env = os.environ.copy()
        env.update(_cli._download_mtp_helper_env())
        self.app.call_from_thread(
            log.write, f"$ ax-engine download-mtp {self.profile.mtp_target}"
        )
        try:
            proc = subprocess.Popen(
                argv,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.app.call_from_thread(self._finish_mtp, None, str(exc))
            return
        assert proc.stdout is not None
        for line in proc.stdout:
            self.app.call_from_thread(log.write, line.rstrip("\n"))
        proc.wait()
        self.app.call_from_thread(self._finish_mtp, proc.returncode, None)

    def _finish_mtp(self, code: int | None, launch_error: str | None) -> None:
        status = self.query_one("#status", Label)
        if launch_error is not None:
            status.update(
                f"[red]Could not launch ax-engine-bench:[/red] {launch_error}   "
                "Press q to quit."
            )
        elif code == 0:
            status.update("[green]MTP package ready.[/green]   Press b for back, q to quit.")
        else:
            status.update(
                f"[red]download-mtp exited {code}.[/red]   Press b for back, q to quit."
            )


class DownloaderApp(App):
    """Minimal model picker that runs a real download on selection."""

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
                    self.push_screen(DownloadScreen(profile, choice))

            self.push_screen(VariantScreen(profile), resolve)
        else:
            self.push_screen(DownloadScreen(profile, "direct"))


def main() -> int:
    DownloaderApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
