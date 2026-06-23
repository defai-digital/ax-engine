"""AX Engine TUI — a Textual-based terminal UI for model management.

Provides two tabs:
  * Downloader — browse the model catalog, download with live progress.
  * Serve — pick an installed model, configure host/port, launch ax-engine-server.

Entry point: `ax-engine tui`.  Requires the optional `textual` dependency:

    pip install ax-engine[tui]
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import get_current_worker

from ._cli import (
    MODEL_PROFILES,
    ModelProfile,
    _default_download_root,
    _download_mtp_helper_env,
    _downloadable_profiles,
    _profile_for_model,
)


# ---------------------------------------------------------------------------
# download_model.py helper (loaded by path — it is a script, not a package)
# ---------------------------------------------------------------------------

def _load_download_model():
    """Import scripts/download_model.py by path."""
    candidates: list[Path] = []
    explicit_root = os.environ.get("AX_ENGINE_REPO_ROOT")
    if explicit_root:
        candidates.append(Path(explicit_root) / "scripts" / "download_model.py")
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidates.append(parent / "scripts" / "download_model.py")
        candidates.append(parent / "download_model.py")
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("ax_dm_tui", candidate)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    return None


_DM = _load_download_model()


def _model_is_installed(profile: ModelProfile) -> bool:
    """Check whether a model's snapshot directory exists in the HF cache."""
    if _DM is None:
        return False
    try:
        repo_dir = _DM.default_mlx_lm_repo_cache_dir(profile.repo_id)
        return repo_dir.exists() and any(repo_dir.iterdir())
    except Exception:
        return False


def _installed_profiles() -> list[ModelProfile]:
    return [p for p in _downloadable_profiles() if _model_is_installed(p)]


def _format_bytes(num: float | None) -> str:
    if num is None:
        return "-- B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

APP_CSS = """
TabbedContent {
    height: 1fr;
}
TabPane {
    padding: 0;
}
DataTable {
    height: 1fr;
}
.status-installed {
    color: $success;
}
.status-missing {
    color: $warning;
}
#serve-controls {
    height: auto;
    padding: 1 2;
    border-top: solid $accent;
}
#serve-controls Horizontal {
    height: auto;
    margin-bottom: 1;
}
#serve-controls Label {
    width: 8;
    content-align: right middle;
    margin-right: 1;
}
#serve-controls Input {
    width: 1fr;
}
#serve-log {
    height: 1fr;
    border-top: solid $accent;
}
#serve-url {
    padding: 0 2;
    color: $success;
}
#download-status {
    padding: 0 2;
    height: auto;
}
#download-detail {
    padding: 0 2;
    height: auto;
}
#download-bar {
    padding: 0 2;
    height: auto;
}
"""


# ---------------------------------------------------------------------------
# Variant picker (Direct vs MTP)
# ---------------------------------------------------------------------------

class VariantScreen(ModalScreen[str | None]):
    """Direct-vs-MTP choice for a model that has an MTP acceleration package."""

    CSS = """
    VariantScreen { align: center middle; }
    #dialog {
        width: 80; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #dialog Label { margin-bottom: 1; }
    #dialog Button { width: 100%; margin-bottom: 1; }
    """

    def __init__(self, profile: ModelProfile) -> None:
        super().__init__()
        self.profile = profile

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"[b]{self.profile.label}[/b] has an MTP acceleration package.")
            yield Label("Download which variant?")
            yield Button(
                f"  Direct download   ({self.profile.repo_id})",
                id="direct",
                variant="primary",
            )
            yield Button(
                f"  MTP package       (download-mtp {self.profile.mtp_target})",
                id="mtp",
                variant="success",
            )
            yield Button("  Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None if event.button.id == "cancel" else event.button.id)


# ---------------------------------------------------------------------------
# Download screen (progress + log)
# ---------------------------------------------------------------------------

class DownloadScreen(Screen):
    """Runs the real download for one model and reports progress live."""

    CSS = """
    DownloadScreen {
        layout: vertical;
    }
    #dl-header {
        height: auto;
        padding: 1 2;
        background: $surface;
    }
    #dl-status { height: auto; padding: 0 2; }
    #dl-bar-container { height: auto; padding: 0 2; }
    #dl-detail { height: auto; padding: 0 2; }
    #dl-log { height: 1fr; }
    """

    BINDINGS = [
        Binding("b", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, profile: ModelProfile, variant: str) -> None:
        super().__init__()
        self.profile = profile
        self.variant = variant
        self._total: int | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="dl-header"):
            yield Label(f"Downloading [b]{self.profile.repo_id}[/b]", id="dl-status")
            if self.variant == "direct":
                with Vertical(id="dl-bar-container"):
                    yield ProgressBar(total=None, show_eta=True, id="dl-bar")
                yield Label("", id="dl-detail")
            else:
                yield RichLog(id="dl-log", markup=False, highlight=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        if self.variant == "direct":
            self.run_direct()
        else:
            self.run_mtp()

    def action_back(self) -> None:
        self.dismiss()

    # -- direct download ----------------------------------------------------

    @work(thread=True, exclusive=True)
    def run_direct(self) -> None:
        worker = get_current_worker()
        if _DM is None:
            self.app.call_from_thread(
                self._finish, RuntimeError("download_model.py helper not found"), 0
            )
            return
        repo = self.profile.repo_id
        repo_dir = _DM.default_mlx_lm_repo_cache_dir(repo)
        try:
            total = _DM._total_repo_bytes(repo)
        except Exception:
            total = None
        self.app.call_from_thread(self._set_total, total)

        err: dict[str, BaseException] = {}

        def _do() -> None:
            try:
                _DM.download(repo, None, quiet=True)
            except BaseException as exc:  # noqa: BLE001
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
            downloaded = _DM._dir_size_bytes(repo_dir)
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
        self.app.call_from_thread(self._finish, err.get("e"), _DM._dir_size_bytes(repo_dir))

    def _set_total(self, total: int | None) -> None:
        self._total = total
        self.query_one(ProgressBar).update(total=total)

    def _update(self, downloaded: int, speed: float | None) -> None:
        if self._total:
            self.query_one(ProgressBar).update(progress=min(downloaded, self._total))
        self.query_one("#dl-detail", Label).update(self._detail(downloaded, speed))

    def _detail(self, downloaded: int, speed: float | None) -> str:
        size = (
            f"{_format_bytes(downloaded)} / {_format_bytes(self._total)}"
            if self._total
            else _format_bytes(downloaded)
        )
        rate = f"{_format_bytes(speed)}/s" if speed else "-- B/s"
        return f"{size}   |   {rate}"

    def _finish(self, err: BaseException | None, downloaded: int) -> None:
        status = self.query_one("#dl-status", Label)
        if err is not None:
            status.update(f"[red]Failed:[/red] {err}")
            return
        if self._total:
            self.query_one(ProgressBar).update(progress=self._total)
        self.query_one("#dl-detail", Label).update(self._detail(downloaded, None))
        status.update("[green]Done.[/green]  Press [b]b[/b] for back, [b]q[/b] to quit.")

    # -- MTP download -------------------------------------------------------

    @work(thread=True, exclusive=True)
    def run_mtp(self) -> None:
        from . import _cli

        log = self.query_one(RichLog)
        argv = [str(_cli._bench_bin()), "download-mtp", self.profile.mtp_target]
        env = os.environ.copy()
        env.update(_download_mtp_helper_env())
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
        status = self.query_one("#dl-status", Label)
        if launch_error is not None:
            status.update(f"[red]Could not launch ax-engine-bench:[/red] {launch_error}")
        elif code == 0:
            status.update("[green]MTP package ready.[/green]  Press [b]b[/b] for back.")
        else:
            status.update(f"[red]download-mtp exited {code}.[/red]  Press [b]b[/b] for back.")


# ---------------------------------------------------------------------------
# Downloader tab
# ---------------------------------------------------------------------------

class DownloaderTab(Static):
    """Model browser with download capability."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "download", "Download"),
    ]

    def __init__(self, force: bool = False) -> None:
        super().__init__()
        self._force = force
        self._profiles = _downloadable_profiles()

    def compose(self) -> ComposeResult:
        yield DataTable(id="dl-table", zebra_stripes=True)
        yield Label(
            f"  Default destination: {_default_download_root()}",
            id="dl-dest",
        )

    def on_mount(self) -> None:
        table = self.query_one("#dl-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "Model", "MTP", "Repo", "Status")
        for index, profile in enumerate(self._profiles, start=1):
            installed = _model_is_installed(profile)
            status_text = "[green]installed[/green]" if installed else "[yellow]--[/yellow]"
            table.add_row(
                str(index),
                profile.label,
                "yes" if profile.mtp_target else "--",
                profile.repo_id,
                status_text,
                key=str(index - 1),
            )
        table.focus()

    def action_download(self) -> None:
        table = self.query_one("#dl-table", DataTable)
        if table.cursor_row is None:
            return
        profile = self._profiles[table.cursor_row]
        if profile.mtp_target:
            def resolve(choice: str | None) -> None:
                if choice is not None:
                    self.app.push_screen(DownloadScreen(profile, choice))

            self.app.push_screen(VariantScreen(profile), resolve)
        else:
            self.app.push_screen(DownloadScreen(profile, "direct"))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        profile = self._profiles[int(event.row_key.value)]
        if profile.mtp_target:
            def resolve(choice: str | None) -> None:
                if choice is not None:
                    self.app.push_screen(DownloadScreen(profile, choice))

            self.app.push_screen(VariantScreen(profile), resolve)
        else:
            self.app.push_screen(DownloadScreen(profile, "direct"))


# ---------------------------------------------------------------------------
# Serve tab
# ---------------------------------------------------------------------------

class ServeTab(Static):
    """Serve launcher — pick an installed model and start ax-engine-server."""

    def __init__(self) -> None:
        super().__init__()
        self._profiles = _downloadable_profiles()
        self._server_proc: subprocess.Popen | None = None

    def compose(self) -> ComposeResult:
        yield Label("  Select an installed model to serve:", id="serve-hint")
        yield DataTable(id="serve-table", zebra_stripes=True)
        with Vertical(id="serve-controls"):
            with Horizontal():
                yield Label("Host:")
                yield Input(value="127.0.0.1", id="serve-host", placeholder="127.0.0.1")
            with Horizontal():
                yield Label("Port:")
                yield Input(value="8080", id="serve-port", placeholder="8080")
            with Horizontal():
                yield Button("Start Server", id="serve-start", variant="success")
                yield Button("Stop Server", id="serve-stop", variant="error")
        yield Label("", id="serve-url")
        yield RichLog(id="serve-log", markup=False, highlight=False, wrap=True)

    def on_mount(self) -> None:
        table = self.query_one("#serve-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "Model", "Preset", "Repo")
        installed = _installed_profiles()
        if not installed:
            self.query_one("#serve-hint", Label).update(
                "  [yellow]No installed models found.[/yellow]  "
                "Download one in the Downloader tab first."
            )
        for index, profile in enumerate(installed, start=1):
            preset = profile.preset or "--"
            table.add_row(
                str(index),
                profile.label,
                preset,
                profile.repo_id,
                key=str(index - 1),
            )
        self._installed = installed

    def _get_selected_profile(self) -> ModelProfile | None:
        table = self.query_one("#serve-table", DataTable)
        if table.cursor_row is None:
            return None
        installed = getattr(self, "_installed", [])
        if 0 <= table.cursor_row < len(installed):
            return installed[table.cursor_row]
        return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "serve-start":
            self._start_server()
        elif event.button.id == "serve-stop":
            self._stop_server()

    def _start_server(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            log = self.query_one("#serve-log", RichLog)
            log.write("[red]No model selected.[/red]")
            return
        if self._server_proc is not None and self._server_proc.poll() is None:
            log = self.query_one("#serve-log", RichLog)
            log.write("[yellow]Server is already running.[/yellow]")
            return

        from . import _cli

        host_input = self.query_one("#serve-host", Input)
        port_input = self.query_one("#serve-port", Input)
        host = host_input.value.strip() or "127.0.0.1"
        port = port_input.value.strip() or "8080"

        server_bin = str(_cli._server_bin())
        argv = [server_bin, "--host", host, "--port", port, "--mlx"]

        if profile.preset is not None:
            argv.extend(["--preset", profile.preset, "--resolve-model-artifacts", "hf-cache"])
        else:
            repo_dir = None
            if _DM is not None:
                try:
                    repo_dir = _DM.default_mlx_lm_repo_cache_dir(profile.repo_id)
                except Exception:
                    pass
            if repo_dir is not None:
                argv.extend(["--mlx-model-artifacts-dir", str(repo_dir)])

        log = self.query_one("#serve-log", RichLog)
        log.write(f"$ {' '.join(argv)}")

        try:
            self._server_proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.stream_server_output()
            url = f"http://{host}:{port}"
            self.query_one("#serve-url", Label).update(f"  Server running at: {url}")
        except OSError as exc:
            log.write(f"[red]Failed to start server:[/red] {exc}")

    def _stop_server(self) -> None:
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.terminate()
            log = self.query_one("#serve-log", RichLog)
            log.write("[yellow]Server stopped.[/yellow]")
            self.query_one("#serve-url", Label).update("")
        self._server_proc = None

    @work(thread=True, exclusive=True)
    def stream_server_output(self) -> None:
        proc = self._server_proc
        if proc is None or proc.stdout is None:
            return
        log = self.query_one("#serve-log", RichLog)
        for line in proc.stdout:
            self.app.call_from_thread(log.write, line.rstrip("\n"))
        proc.wait()
        code = proc.returncode
        if code != 0:
            self.app.call_from_thread(
                log.write, f"[red]Server exited with code {code}.[/red]"
            )
            self.app.call_from_thread(
                self.query_one.__wrapped__, "#serve-url", Label
            ) if False else None
        self._server_proc = None


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class AxTuiApp(App):
    """AX Engine TUI — model downloader and serve launcher."""

    TITLE = "AX Engine"
    CSS = APP_CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "next_tab", "Next Tab"),
    ]

    def __init__(self, force: bool = False) -> None:
        super().__init__()
        self._force = force

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="downloader"):
            with TabPane("Downloader", id="downloader"):
                yield DownloaderTab(force=self._force)
            with TabPane("Serve", id="serve"):
                yield ServeTab()
        yield Footer()

    def action_next_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        panes = list(tabs.panes)
        current = tabs.active
        for i, pane in enumerate(panes):
            if pane.id == current:
                next_pane = panes[(i + 1) % len(panes)]
                tabs.active = next_pane.id
                break


def run(force: bool = False) -> int:
    """Launch the TUI application."""
    AxTuiApp(force=force).run()
    return 0
