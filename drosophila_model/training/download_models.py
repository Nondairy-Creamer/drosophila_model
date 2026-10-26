'''
Rsync utility for transferring model files to/from a remote server.

Usage (from the repo root):

    python -m drosophila_model.training.download_models <server>
    python -m drosophila_model.training.download_models <server> --model-folder <run>
    python -m drosophila_model.training.download_models <server> --push --model-folder <run>

Uses rsync over ssh for resumable, high-throughput transfers. Requires
rsync on both ends (preinstalled on macOS and most Linux containers; it is
installed on the remote with apt-get if missing).

Pulling everything groups runs locally by tag/timestamp:
``models/<tag>/<tag>_gpuN_runM/`` (``find_best_model`` scans that layout);
folders without a ``_gpuN_runM`` suffix land flat in ``models/<name>/``.
'''

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict

_LOCAL_MODELS_DIR = Path('models')  # relative to the working directory (the repo root)
_DEFAULT_REMOTE_MODELS_DIR = '/workspace/drosophila_model/models'
_SSH_CONFIG_PATH = Path.home() / '.ssh' / 'config'
_SSH_KEY_PATH = '~/.ssh/id_ed25519'


def _parse_ssh_config(config_path: Path = _SSH_CONFIG_PATH) -> Dict[str, Dict[str, str]]:
    '''
    Parse an OpenSSH config file and return a dict mapping Host names to their
    settings (HostName, Port, User, IdentityFile).
    '''
    hosts: Dict[str, Dict[str, str]] = {}
    if not config_path.exists():
        return hosts

    current_host = None
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = re.match(r'^(\S+)\s+(.+)$', line)
            if not match:
                continue
            key, value = match.group(1), match.group(2)
            if key.lower() == 'host':
                current_host = value
                hosts[current_host] = {}
            elif current_host is not None:
                hosts[current_host][key.lower()] = value
    return hosts


def _resolve_server(server: str) -> tuple[str, str, str, str]:
    '''
    Resolve a server argument to (hostname, port, user, identity_file).

    *server* can be:
      - An IP:port string, e.g. "194.68.245.60:22050"
      - A Host name from ~/.ssh/config, e.g. "RunPodServer1"
    '''
    # Try IP:port format first
    if re.match(r'^\d+\.\d+\.\d+\.\d+:\d+$', server):
        ip, port = server.rsplit(':', 1)
        return ip, port, 'root', _SSH_KEY_PATH

    # Otherwise look up in SSH config
    hosts = _parse_ssh_config()
    if server not in hosts:
        raise ValueError(
            f"Server '{server}' not found in SSH config ({_SSH_CONFIG_PATH}) "
            f"and is not a valid IP:port. Available hosts: {list(hosts.keys())}"
        )
    entry = hosts[server]
    return (
        entry.get('hostname', ''),
        entry.get('port', '22'),
        entry.get('user', 'root'),
        entry.get('identityfile', _SSH_KEY_PATH),
    )


def _build_ssh_arg(port: str, identity_file: str) -> str:
    '''
    Build the `-e` argument for rsync that invokes ssh with throughput-tuned
    options. Disables compression (model files are already binary/compressed)
    and tags the connection for throughput-optimized routing.
    '''
    return (
        f'ssh -p {port} -i {identity_file} '
        '-o Compression=no '
        '-o IPQoS=throughput '
        '-o ServerAliveInterval=30'
    )


def _rsync_base(port: str, identity_file: str) -> list[str]:
    '''
    Standard rsync invocation:
      -a  archive (preserve perms/times)
      -h  human-readable sizes
      -P  --partial + --progress (resumable + live progress)
      --inplace  write directly to destination file (faster, less disk churn,
                 and resumes cleanly mid-file when combined with --partial)
      --no-o --no-g  don't preserve owner/group. Container filesystems (e.g.
                 RunPod) reject chown, which otherwise makes rsync exit
                 non-zero even though the data transferred fine.
    '''
    return [
        'rsync', '-ahP', '--inplace', '--no-o', '--no-g',
        '-e', _build_ssh_arg(port, identity_file),
    ]


def _ensure_remote_rsync(ssh_base: list[str]) -> None:
    '''
    Make sure rsync exists on the remote. RunPod containers often ship without
    it, which makes rsync-over-ssh fail with "rsync: command not found"
    (exit 127). If missing, try to install it via apt-get.
    '''
    check = subprocess.run(
        ssh_base + ['command -v rsync'],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        return

    print('rsync not found on remote; attempting to install it...')
    install_cmd = (
        'apt-get update && apt-get install -y rsync'
    )
    result = subprocess.run(ssh_base + [install_cmd])
    if result.returncode != 0:
        raise RuntimeError(
            'Failed to install rsync on the remote. Install it manually '
            '(e.g. `apt-get update && apt-get install -y rsync`) and retry.'
        )


def rsync_transfer(
    server: str,
    transfer_type: str = 'pull',
    folder: str | None = None,
    remote_models_dir: str = _DEFAULT_REMOTE_MODELS_DIR,
) -> None:
    '''
    Rsync files to or from a remote server.

    Args:
        server: Either an IP:port string (e.g. "194.68.245.60:22050") or an
                SSH config host name (e.g. "RunPodServer1").
        transfer_type: "pull" to download from remote, "push" to upload.
        folder: Optional name of a specific model folder. If omitted,
                the entire models directory is transferred.
        remote_models_dir: The remote repo's models directory.
    '''
    hostname, port, user, identity_file = _resolve_server(server)
    remote_models_dir = remote_models_dir.rstrip('/')

    # Strip leading "models/" if the user accidentally includes it,
    # since _LOCAL_MODELS_DIR / remote_models_dir already point to the models dir.
    if folder:
        folder = folder.removeprefix('models/')
        folder = folder.strip('/')

    if folder:
        remote_path = f'{remote_models_dir}/{folder}/'
        local_path = str(_LOCAL_MODELS_DIR / folder)
    else:
        remote_path = f'{remote_models_dir}/'
        local_path = str(_LOCAL_MODELS_DIR) + '/'

    # Check before contacting the remote (a push would otherwise create an
    # empty remote folder first).
    if transfer_type == 'push' and not Path(local_path).is_dir():
        raise FileNotFoundError(f"Local folder to push not found: {Path(local_path)}")

    if shutil.which('rsync') is None:
        raise RuntimeError(
            'rsync not found on this machine. Install it locally (it ships with '
            'macOS and most Linux distributions; on Windows run this from WSL) '
            'and retry.'
        )

    ssh_base = ['ssh', '-p', port, '-i', identity_file, f'{user}@{hostname}']
    rsync_base = _rsync_base(port, identity_file)

    _ensure_remote_rsync(ssh_base)

    if transfer_type == 'pull':
        if folder:
            os.makedirs(local_path, exist_ok=True)
            # Pull the folder's top-level files plus the training loss plot
            # (figures/training/); the final `- */` filter excludes every
            # other subdirectory (-a recurses otherwise).
            remote_spec = f'{user}@{hostname}:{remote_path}'
            cmd = rsync_base + [
                '-f', '+ /figures/', '-f', '+ /figures/training/', '-f', '- */',
                remote_spec, local_path + '/',
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("Transfer complete.")
            return

        # No folder specified: list remote subdirs and group by tag/timestamp
        # prefix into models/<prefix>/. Folders without a _gpuN_runM suffix
        # (prefix == name) are pulled flat into models/ instead of nesting
        # them in a folder of their own name.
        list_cmd = ssh_base + [f'ls -1 {remote_path}']
        result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
        remote_entries = [
            e.strip().rstrip('/') for e in result.stdout.strip().split('\n') if e.strip()
        ]
        if not remote_entries:
            print("No entries found in remote models directory.")
            return

        groups: dict[str, list[str]] = {}
        for entry in remote_entries:
            prefix = re.sub(r'_gpu\d+_run\d+$', '', entry)
            groups.setdefault(prefix, []).append(entry)

        for prefix, entries in groups.items():
            for entry in entries:
                local_dir = _LOCAL_MODELS_DIR if entry == prefix else _LOCAL_MODELS_DIR / prefix
                os.makedirs(local_dir, exist_ok=True)
                remote_spec = f'{user}@{hostname}:{remote_models_dir}/{entry}'
                cmd = rsync_base + [remote_spec, str(local_dir) + '/']
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

        print("Transfer complete.")
        return

    if transfer_type == 'push':
        # Ensure remote destination directory exists
        mkdir_cmd = ssh_base + [f'mkdir -p {remote_path}']
        print(f"Ensuring remote directory exists: {remote_path}")
        subprocess.run(mkdir_cmd, check=True)

        if folder:
            local_dir = Path(local_path)
            files = sorted(str(f) for f in local_dir.iterdir() if f.is_file())
            if not files:
                print("No files found in local directory.")
                return
            remote_spec = f'{user}@{hostname}:{remote_path}'
            cmd = rsync_base + files + [remote_spec]
        else:
            remote_spec = f'{user}@{hostname}:{remote_path}'
            # Trailing slash on local_path makes rsync copy the *contents*
            # of the models dir into the remote models dir.
            cmd = rsync_base + [local_path, remote_spec]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("Transfer complete.")
        return

    raise ValueError(f"transfer_type must be 'pull' or 'push', got '{transfer_type}'")


def main() -> None:
    '''Entry point for the rsync CLI.'''
    parser = argparse.ArgumentParser(
        description='Rsync model folders to/from a remote server.'
    )
    parser.add_argument(
        'server',
        help=(
            'Server to connect to. Either IP:port (e.g. 194.68.245.60:22050) '
            'or an SSH config host name (e.g. RunPodServer1).'
        ),
    )
    parser.add_argument(
        '--push',
        action='store_true',
        help='Upload to the remote. If omitted, pulls (downloads) instead.',
    )
    parser.add_argument(
        '--model-folder',
        default=None,
        help='Name of a specific model folder to transfer. If omitted, transfers all.',
    )
    parser.add_argument(
        '--remote-dir',
        default=_DEFAULT_REMOTE_MODELS_DIR,
        help=f'Models directory on the remote (default: {_DEFAULT_REMOTE_MODELS_DIR}).',
    )

    args = parser.parse_args()
    rsync_transfer(
        server=args.server,
        transfer_type='push' if args.push else 'pull',
        folder=args.model_folder,
        remote_models_dir=args.remote_dir,
    )


if __name__ == '__main__':
    main()
