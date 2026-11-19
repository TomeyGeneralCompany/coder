"""Install a normal Linux application-menu/desktop shortcut for local Coder.

Usage (from software-agent-sdk-main):
    uv run python -m coder_local.desktop

No sudo, system service, or global package installation is required. The
launcher invokes the existing checkout's uv environment and opens the browser.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


def install_launcher(home: Path | None = None, checkout: Path | None = None) -> dict[str, str]:
    home = (home or Path.home()).expanduser().resolve()
    checkout = (checkout or Path(__file__).resolve().parent.parent).resolve()
    if not (checkout / "coder_local" / "server.py").is_file():
        raise ValueError("Coder source checkout is missing")
    bin_dir = home / ".local" / "bin"
    applications = home / ".local" / "share" / "applications"
    bin_dir.mkdir(parents=True, exist_ok=True)
    applications.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "coder-local-studio"
    script = (
        "#!/bin/sh\n"
        "set -eu\n"
        'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"\n'
        f"cd {shlex.quote(str(checkout))}\n"
        'if command -v curl >/dev/null 2>&1 && '
        'curl --silent --fail --max-time 1 http://127.0.0.1:8765/health >/dev/null; then\n'
        '  exec python3 -m webbrowser http://127.0.0.1:8765\n'
        "fi\n"
        "exec uv run --no-sync python -m coder_local.server --open-browser\n"
    )
    wrapper.write_text(script, encoding="utf-8")
    wrapper.chmod(0o755)
    entry = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        "Name=Coder\n"
        "GenericName=Local AI Development Studio\n"
        "Comment=Work with local source repositories using Qwen3:4b\n"
        f'Exec="{str(wrapper).replace(chr(34), chr(92)+chr(34))}"\n'
        "Icon=applications-development\n"
        "Terminal=false\n"
        "StartupNotify=true\n"
        "Categories=Development;IDE;\n"
        "Keywords=Coder;OpenHands;Ollama;Qwen;Git;\n"
    )
    application_entry = applications / "coder-local.desktop"
    application_entry.write_text(entry, encoding="utf-8")
    application_entry.chmod(0o755)

    result = {"launcher": str(application_entry), "command": str(wrapper)}
    desktop = home / "Desktop"
    if desktop.is_dir():
        desktop_entry = desktop / "Coder.desktop"
        desktop_entry.write_text(entry, encoding="utf-8")
        desktop_entry.chmod(0o755)
        result["desktop_icon"] = str(desktop_entry)
        gio = shutil.which("gio")
        if gio:
            subprocess.run(
                [gio, "set", str(desktop_entry), "metadata::trusted", "true"],
                timeout=5, check=False, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    return result


def main() -> None:
    details = install_launcher()
    print("Coder launcher installed:")
    for label, value in details.items():
        print(f"  {label}: {value}")
    print("Open Coder from your application menu or desktop shortcut.")


if __name__ == "__main__":
    main()
