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
# A folder untouched for this long is swept. A reload does not touch the
# folders: they re-attach when the conversation resumes.
WORKSPACE_MAX_AGE = 7 * 24 * 3600


class Workspace:
    """One folder per session: agent-written files and downloaded attachments.

    Paths are confined to the session folder: absolute paths and .. are
    refused. The folder outlives the RAM session (a reload must not take
    the files with it) and dies on the explicit session drops and on the
    age sweep.
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
        """The confined target of a workspace path, or None when it escapes."""
        base = self.folder(session_id).resolve()
        try:
            target = (base / path).resolve()
        except OSError:
            return None
        if target != base and base not in target.parents:
            return None
        return target

    def session_size(self, session_id: int) -> int:
        """The total size of the files of one session."""
        folder = self.folder(session_id)
        try:
            return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        except OSError:
            return 0

    def touch(self, session_id: int) -> None:
        """Refresh the folder time: the age sweep reads it."""
        with contextlib.suppress(OSError):
            os.utime(self.folder(session_id))

    def drop(self, session_id: int) -> None:
        """Delete the folder of one session."""
        shutil.rmtree(self.folder(session_id), ignore_errors=True)

    def drop_all(self) -> None:
        """Delete the whole workspace root."""
        shutil.rmtree(self.root, ignore_errors=True)

    def sweep(self) -> None:
        """The janitor: delete the folders untouched for WORKSPACE_MAX_AGE."""
        try:
            folders = list(self.root.iterdir())
        except OSError:
            return
        cutoff = time.time() - WORKSPACE_MAX_AGE
        for folder in folders:
            try:
                if folder.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue
            shutil.rmtree(folder, ignore_errors=True)
            log.info("The workspace folder %s was swept: untouched past the age cap.", folder.name)
