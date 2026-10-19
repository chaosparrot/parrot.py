import sys


def main():
    from importlib.util import find_spec
    if find_spec("PyQt6") is None:
        sys.exit("The GUI needs PyQt6: pip install -r requirements-gui.txt")

    from gui.app import create_app
    app = create_app(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
