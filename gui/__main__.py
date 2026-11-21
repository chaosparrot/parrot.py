import sys


def _preload_torch():
    """Windows only. PyQt6 bundles its own msvcp140.dll. If Qt loads first,
    torch gets that copy and fails (WinError 1114) whenever it was built
    against a newer one. The Models tab reads checkpoints with torch.

    About a second at launch. If torch won't load at all, models show as
    unreadable."""
    if sys.platform == "win32":
        try:
            import torch  # noqa: F401
        except Exception:
            pass


def main():
    from importlib.util import find_spec
    missing = [m for m in ("PyQt6", "pyqtgraph") if find_spec(m) is None]
    if missing:
        sys.exit(f"The GUI needs {' and '.join(missing)}: "
                 "pip install -r requirements-gui.txt")

    _preload_torch()
    from gui.app import create_app
    app = create_app(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
