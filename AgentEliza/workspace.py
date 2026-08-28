"""The agent workspace: one folder per conversation session in the OS temp dir."""

import contextlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

log = logging.getLogger("red.agenteliza.workspace")

# Caps of the workspace: one file and one session folder. An attachment of
# any practical size fits, and the few live sessions keep the worst-case
# footprint modest.
WORKSPACE_FILE_MAX_BYTES = 26_214_400  # 25 MiB, the DM upload limit
WORKSPACE_SESSION_MAX_BYTES = 262_144_000  # 250 MiB
# A file untouched for this long is swept. A reload does not touch the
# files: they re-attach when the conversation resumes.
WORKSPACE_MAX_AGE = 30 * 24 * 3600
# The sweep runs about once a day: the filesystem does not need a scan on
# every minute of the compaction sweeper.
WORKSPACE_SWEEP_INTERVAL = 24 * 3600


class Workspace:
    """One folder per session: agent-written files and downloaded attachments.

    Paths are confined to the session folder: absolute paths and .. are
    refused. The folder outlives the RAM session (a reload must not take
    the files with it) and dies on the explicit session drops and when
    the age sweep empties it.
    """

    def __init__(self):
        self.root = Path(tempfile.gettempdir()) / "agenteliza"
        with contextlib.suppress(OSError):
            self.root.mkdir(mode=0o700, exist_ok=True)

    def folder(self, session_id: int) -> Path:
        return self.root / str(session_id)

    def make_folder(self, session_id: int) -> Path:
        """The session folder, created private on first use."""
        folder = self.folder(session_id)
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        return folder

    def resolve(self, session_id: int, path: str) -> Path | None:
        """The confined target of a workspace path, or None when a symlink points outside.

        The path is unix-style and rooted at the session folder: a leading
        / names the workspace root, so /file.txt is the file file.txt of
        the workspace, and a backslash reads as a separator. Like at the
        root of a unix filesystem, .. past the root stays at the root:
        /../file.txt is /file.txt.
        """
        parts = []
        for part in path.replace("\\", "/").split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        base = self.folder(session_id).resolve()
        try:
            target = base.joinpath(*parts).resolve()
        except OSError:
            return None
        if target != base and base not in target.parents:
            # A link inside the workspace points outside it.
            return None
        return target

    def relpath(self, session_id: int, target) -> str:
        """The agent-facing path of a confined target: unix-style, rooted at the workspace folder."""
        return "/" + target.relative_to(self.folder(session_id).resolve()).as_posix()

    def session_size(self, session_id: int) -> int:
        """The total size of the files of one session."""
        folder = self.folder(session_id)
        try:
            return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        except OSError:
            return 0

    def touch(self, path) -> None:
        """Refresh the usage time of one file: the age sweep reads it."""
        with contextlib.suppress(OSError):
            os.utime(path)

    def drop(self, session_id: int) -> None:
        """Delete the folder of one session."""
        shutil.rmtree(self.folder(session_id), ignore_errors=True)

    def drop_all(self) -> None:
        """Delete the whole workspace root."""
        shutil.rmtree(self.root, ignore_errors=True)

    def sweep(self) -> None:
        """The janitor: delete the files untouched for WORKSPACE_MAX_AGE, then the folders left empty."""
        try:
            folders = list(self.root.iterdir())
        except OSError:
            return
        cutoff = time.time() - WORKSPACE_MAX_AGE
        for folder in folders:
            try:
                if not folder.is_dir():
                    continue
                swept = 0
                for path in folder.rglob("*"):
                    try:
                        if not path.is_file() or path.stat().st_mtime >= cutoff:
                            continue
                        path.unlink()
                        swept += 1
                    except OSError:
                        continue
                if swept:
                    log.info("Swept %d workspace file(s) of session %s: untouched past the age cap.", swept, folder.name)
                # The folders the sweep leaves empty die with their files, deepest first.
                for path in sorted((p for p in folder.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                    with contextlib.suppress(OSError):
                        path.rmdir()
                with contextlib.suppress(OSError):
                    folder.rmdir()
            except OSError:
                continue
