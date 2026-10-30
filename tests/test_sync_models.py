import os

import pytest

from drosophila_model.training import sync_models


def test_cygwin_path():
    assert sync_models._cygwin_path(r'C:\Users\me\.ssh\id_rsa') == '/cygdrive/c/Users/me/.ssh/id_rsa'


def test_ssh_arg_without_bundled_ssh():
    # Linux / macOS / WSL: the plain ssh on PATH, paths passed through as given.
    assert sync_models._build_ssh_arg('22', '~/.ssh/id_ed25519') == (
        'ssh -p 22 -i ~/.ssh/id_ed25519 '
        '-o Compression=no -o IPQoS=throughput -o ServerAliveInterval=30'
    )


@pytest.mark.skipif(os.name == 'nt', reason='checks the non-Windows path')
def test_no_bundled_ssh_off_windows():
    assert sync_models._bundled_ssh() is None


@pytest.mark.skipif(os.name != 'nt', reason='cwRsync is Windows only')
def test_bundled_ssh_behind_chocolatey_shim(tmp_path, monkeypatch):
    shim = tmp_path / 'chocolatey' / 'bin' / 'rsync.exe'
    ssh = tmp_path / 'chocolatey' / 'lib' / 'rsync' / 'tools' / 'bin' / 'ssh.exe'
    for f in (shim, ssh):
        f.parent.mkdir(parents=True)
        f.touch()
    monkeypatch.setattr(sync_models.shutil, 'which', lambda name: str(shim))
    assert sync_models._bundled_ssh() == ssh

    arg = sync_models._build_ssh_arg('22055', r'C:\keys\id_rsa', ssh)
    assert arg.startswith(f'{sync_models._cygwin_path(ssh)} -p 22055 -i /cygdrive/c/keys/id_rsa ')
    assert '-o UserKnownHostsFile=/cygdrive/' in arg
