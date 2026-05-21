#!/usr/bin/env python3
"""Set --global git user.email / user.name from config.toml's [user] section.

Run once per server / workstation after cloning the repo and filling in your
own email + name in config.toml (gitignored). Errors out if either field is
blank so we don't accidentally commit as the default-config "nobody".

Usage:
    python register_git_on_server.py
"""
import os
import subprocess
import sys
from pathlib import Path

from drosophila_model.utilities import data_utilities as du


def _set_git_global(key: str, value: str) -> None:
    subprocess.run(['git', 'config', '--global', key, value], check=True)


def main() -> int:
    # Resolve config relative to this script (the repo root) rather than the
    # current working directory, so the script works from anywhere.
    os.chdir(Path(__file__).resolve().parent)
    config = du.load_config()
    user = config.get('user', {})
    email = (user.get('email') or '').strip()
    name = (user.get('name') or '').strip()

    missing = [k for k, v in (('email', email), ('name', name)) if not v]
    if missing:
        sys.stderr.write(
            f"register_git_on_server: [user] {' and '.join(missing)} not set "
            f"in config.toml. Fill in the [user] section of your local "
            f"config.toml (gitignored) and re-run.\n"
        )
        return 1

    _set_git_global('user.email', email)
    _set_git_global('user.name', name)
    print(f"git --global user.email = {email}")
    print(f"git --global user.name  = {name}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
