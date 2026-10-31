import sys


def main():
    from importlib.util import find_spec
    missing = [m for m in ("PyQt6", "pyqtgraph") if find_spec(m) is None]
    if missing:
        sys.exit(f"The GUI needs {' and '.join(missing)}: "
                 "pip install -r requirements-gui.txt")

    from gui.app import create_app
    app = create_app(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
