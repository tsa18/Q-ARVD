"""
snapshot.py —— PyTorch 实验快照工具

用法：
    from snapshot import Snapshot

    snap = Snapshot(
        command="python train.py --lr 0.01 --epochs 50",
        save_dir="snapshots/exp_001",   # 可选，默认 snapshots/<时间戳>
    )
    snap_path = snap.save()             # 返回快照目录 Path
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
    参数
    ----
    command : str
        本次实验命令，仅用于记录。
    save_dir : str | Path, 可选
        快照保存目录。默认为 <项目根>/snapshots/<时间戳>。
    root : str | Path, 可选
        项目根目录，默认为当前工作目录。
    save_env : bool
        是否保存 pip freeze，默认 True。
    verbose : bool
        是否打印日志，默认 True。
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

        # save_dir 本身不应被当作源码目录扫描
        exclude = DEFAULT_EXCLUDE_DIRS | {self._dest.name}
        self._exclude = exclude

    @property
    def path(self) -> Path:
        """快照目录路径（save() 调用前即可获取，用于提前配置 logger 等）。"""
        return self._dest

    def save(self) -> Path:
        """执行快照，返回快照目录 Path。"""
        self._dest.mkdir(parents=True, exist_ok=True)
        self._log(f"\n📸 快照目录：{self._dest}")

        py_files = self._collect_py_files()
        self._copy_sources(py_files)
        self._write_meta(py_files)
        if self.save_env:
            self._save_env_info()

        self._log(f"✅ 快照完成\n")
        return self._dest

    # ── 内部方法 ─────────────────────────────────────────────────────────────

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
        self._log(f"  ✔ 已复制 {len(py_files)} 个 .py 文件 → src/")

    def _write_meta(self, py_files: list) -> None:
        lines = [
            f"快照时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"运行命令：{self.command or '（未指定）'}",
            f"工作目录：{self.root}",
            f"操作系统：{platform.platform()}",
            f"Python：{sys.version.split()[0]}",
            f"文件数：{len(py_files)} 个 .py",
            "", "── 文件列表 ──",
        ] + [f"  {f.relative_to(self.root)}" for f in sorted(py_files)]
        (self._dest / "meta.txt").write_text("\n".join(lines), encoding="utf-8")
        self._log(f"  ✔ 元信息已保存 → meta.txt")

    def _save_env_info(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True,
            )
            (self._dest / "environment.txt").write_text(result.stdout, encoding="utf-8")
            self._log(f"  ✔ 环境信息已保存 → environment.txt")
        except Exception as e:
            self._log(f"  ⚠ 无法获取环境信息：{e}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
