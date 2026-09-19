#!/usr/bin/env python3
"""TowTrace launcher and environment repair utility."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "backend/main.py",
    "backend/cv/tow_cv_only.py",
    "backend/requirements.txt",
    "frontend/index.html",
    "frontend/detection.html",
    "frontend/evaluation.html",
)
REQUIRED_IMPORTS = (
    "fastapi",
    "uvicorn",
    "cv2",
    "numpy",
    "sqlmodel",
    "multipart",
    "fast_alpr",
)


def missing_project_files(app_dir: Path = APP_DIR) -> list[str]:
    return [name for name in REQUIRED_FILES if not (app_dir / name).is_file()]


def project_files_ready(app_dir: Path = APP_DIR) -> bool:
    return not missing_project_files(app_dir)


def venv_python_path(app_dir: Path = APP_DIR, os_name: str = os.name) -> Path:
    if os_name == "nt":
        return app_dir / ".venv" / "Scripts" / "python.exe"
    return app_dir / ".venv" / "bin" / "python"


def run(command: list[str], app_dir: Path = APP_DIR) -> int:
    return subprocess.run(command, cwd=app_dir, check=False).returncode


def dependencies_ready(python_path: Path, app_dir: Path = APP_DIR) -> bool:
    imports = "; ".join(f"import {name}" for name in REQUIRED_IMPORTS)
    return run([str(python_path), "-c", imports], app_dir) == 0


def prepare_environment(app_dir: Path = APP_DIR) -> Path | None:
    python_path = venv_python_path(app_dir)

    if not python_path.is_file():
        print("Creating TowTrace's Python environment...")
        command = [sys.executable, "-m", "venv"]
        if (app_dir / ".venv").exists():
            command.append("--clear")
        command.append(str(app_dir / ".venv"))
        if run(command, app_dir) != 0:
            print("Could not create the Python environment.")
            return None

    if not dependencies_ready(python_path, app_dir):
        print("Installing or repairing TowTrace dependencies...")
        if run(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(app_dir / "backend" / "requirements.txt"),
            ],
            app_dir,
        ) != 0:
            print("Dependency installation failed. Check your internet connection and try again.")
            return None

    if not dependencies_ready(python_path, app_dir):
        print("The environment is still missing a required dependency.")
        return None

    return python_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the TowTrace demo.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Check that all project files are present, then exit.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create/repair the environment without starting the server.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser window.",
    )
    args = parser.parse_args()

    missing = missing_project_files()
    if missing:
        print("Missing required TowTrace files:")
        for name in missing:
            print(f"  - {name}")
        return 1

    if args.diagnose:
        print("TowTrace project files are ready.")
        return 0

    python_path = prepare_environment()
    if python_path is None:
        return 1

    if args.prepare_only:
        print("TowTrace environment is ready.")
        return 0

    url = "http://127.0.0.1:8000"
    print()
    print(f"TowTrace is starting at {url}")
    print("Leave this window open while using the app. Press Ctrl+C to stop it.")

    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    try:
        return run(
            [
                str(python_path),
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]
        )
    except KeyboardInterrupt:
        print("\nTowTrace stopped.")
        return 0



if __name__ == "__main__":
    raise SystemExit(main())
