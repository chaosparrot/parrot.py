"""Qt-free filesystem helpers."""
import os
import sys
import shutil
import subprocess


class LibraryOpError(Exception):
    """A user-facing problem performing a library operation."""


def open_in_file_manager(path):
    """Open ``path`` (a directory, or the folder containing a file) in the
    platform's file manager. Best-effort; raises on a hard failure."""
    if not os.path.exists(path):
        raise LibraryOpError("That location no longer exists.")
    target = path if os.path.isdir(path) else os.path.dirname(path)
    abs_target = os.path.abspath(target)
    try:
        if sys.platform == "win32":
            os.startfile(abs_target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", abs_target])
        else:
            # Linux/WSL: try wslview (opens Windows Explorer from WSL) first,
            # then fall back to xdg-open.
            for opener in ("wslview", "xdg-open"):
                if shutil.which(opener):
                    subprocess.Popen([opener, abs_target])
                    break
            else:
                raise LibraryOpError("No file manager found to open the folder.")
    except LibraryOpError:
        raise
    except Exception as exc:
        raise LibraryOpError(f"Couldn't open the folder: {exc}")
