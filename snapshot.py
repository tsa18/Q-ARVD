"""
snapshot.py -- PyTorch experiment snapshot tool

Usage:
    from snapshot import Snapshot

    snap = Snapshot(
        command="python train.py --lr 0.01 --epochs 50",
        save_dir="snapshots/exp_001",   # optional, default snapshots/<timestamp>
    )
    snap_path = snap.save()             # returns the snapshot directory Path
"""

import sys
import shutil
import subprocess
import platform
from datetime import datetime
from pathlib import Path


DEFAULT_EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules",
}


class Snapshot:
    """
    Parameters
    ----------
    command : str
        The experiment command, for record-keeping only.
    save_dir : str | Path, optional
        Snapshot save directory. Defaults to <project_root>/snapshots/<timestamp>.
    root : str | Path, optional
        Project root directory, defaults to current working directory.
    save_env : bool
        Whether to save pip freeze output, default True.
    verbose : bool
        Whether to print log messages, default True.
    """

    def __init__(
        self,
        command: str = "",
        save_dir=None,
        root=None,
        save_env: bool = True,
        verbose: bool = True,
    ):
        self.command  = command
        self.root     = Path(root).resolve() if root else Path.cwd()
        self.save_env = save_env
        self.verbose  = verbose

        if save_dir:
            self._dest = Path(save_dir).resolve()
        else:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._dest = self.root / "snapshots" / timestamp

        # save_dir itself should not be scanned as source directory
        exclude = DEFAULT_EXCLUDE_DIRS | {self._dest.name}
        self._exclude = exclude

    @property
    def path(self) -> Path:
        """Snapshot directory path (available before save() is called, useful for early logger setup)."""
        return self._dest

    def save(self) -> Path:
        """Execute snapshot, return the snapshot directory Path."""
        self._dest.mkdir(parents=True, exist_ok=True)
        self._log(f"\nSnapshot directory: {self._dest}")

        py_files = self._collect_py_files()
        self._copy_sources(py_files)
        self._write_meta(py_files)
        if self.save_env:
            self._save_env_info()

        self._log(f"Snapshot saved successfully\n")
        return self._dest

    # -- Internal methods ------------------------------------------------------------

    def _collect_py_files(self) -> list:
        files = []
        for path in self.root.rglob("*.py"):
            parts = set(path.relative_to(self.root).parts)
            if parts & self._exclude:
                continue
            files.append(path)
        return files

    def _copy_sources(self, py_files: list) -> None:
        src_dir = self._dest / "src"
        src_dir.mkdir(exist_ok=True)
        for src in py_files:
            target = src_dir / src.relative_to(self.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        self._log(f"  Copied {len(py_files)} .py files -> src/")

    def _write_meta(self, py_files: list) -> None:
        lines = [
            f"Snapshot time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Command: {self.command or '(not specified)'}",
            f"Working directory: {self.root}",
            f"OS: {platform.platform()}",
            f"Python: {sys.version.split()[0]}",
            f"Files: {len(py_files)} .py",
            "", "-- File list --",
        ] + [f"  {f.relative_to(self.root)}" for f in sorted(py_files)]
        (self._dest / "meta.txt").write_text("\n".join(lines), encoding="utf-8")
        self._log(f"  Metadata saved -> meta.txt")

    def _save_env_info(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True,
            )
            (self._dest / "environment.txt").write_text(result.stdout, encoding="utf-8")
            self._log(f"  Environment info saved -> environment.txt")
        except Exception as e:
            self._log(f"  Could not fetch environment info: {e}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
