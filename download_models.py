'''
SCP utility for transferring model files to/from a remote server.
'''

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Dict

_LOCAL_MODELS_DIR = Path(__file__).resolve().parent / 'models'
_REMOTE_MODELS_DIR = '/workspace/drosophila_model/models'
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


def scp_transfer(
    server: str,
    transfer_type: str = 'pull',
    folder: str | None = None,
) -> None:
    '''
    SCP files to or from a remote server.

    Args:
        server: Either an IP:port string (e.g. "194.68.245.60:22050") or an
                SSH config host name (e.g. "RunPodServer1").
        transfer_type: "pull" to download from remote, "push" to upload.
        folder: Optional name of a specific model folder. If omitted,
                the entire models directory is transferred.
    '''
    hostname, port, user, identity_file = _resolve_server(server)

    # Strip leading "models/" if the user accidentally includes it,
    # since _LOCAL_MODELS_DIR / _REMOTE_MODELS_DIR already point to the models dir.
    if folder:
        folder = folder.removeprefix('models/')
        folder = folder.strip('/')

    if folder:
        remote_path = f'{_REMOTE_MODELS_DIR}/{folder}/'
        local_path = str(_LOCAL_MODELS_DIR / folder)
    else:
        remote_path = f'{_REMOTE_MODELS_DIR}/'
        local_path = str(_LOCAL_MODELS_DIR) + '/'

    ssh_base = ['ssh', '-p', port, '-i', identity_file, f'{user}@{hostname}']
    scp_base = ['scp', '-P', port, '-i', identity_file]

    if transfer_type == 'pull':
        if folder:
            os.makedirs(local_path, exist_ok=True)
            # List only files (not subdirectories) on the remote
            list_cmd = ssh_base + [f'find {remote_path} -maxdepth 1 -type f']
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            remote_files = [f for f in result.stdout.strip().split('\n') if f]
            if not remote_files:
                print("No files found in remote directory.")
                return
            sources = [f'{user}@{hostname}:{f}' for f in remote_files]
            cmd = scp_base + sources + [local_path + '/']
        else:
            # List remote subdirectories and group by timestamp prefix
            list_cmd = ssh_base + [f'ls -1 {remote_path}']
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            remote_entries = [
                e.strip().rstrip('/') for e in result.stdout.strip().split('\n') if e.strip()
            ]
            if not remote_entries:
                print("No entries found in remote models directory.")
                return

            # Group by timestamp prefix (YYYYMMDD_HHMMSS)
            groups: dict[str, list[str]] = {}
            for entry in remote_entries:
                match = re.match(r'(\d{8}_\d{6})', entry)
                ts = match.group(1) if match else 'other'
                groups.setdefault(ts, []).append(entry)

            for ts, entries in groups.items():
                local_ts_dir = _LOCAL_MODELS_DIR / ts
                os.makedirs(local_ts_dir, exist_ok=True)
                print(f"Pulling {len(entries)} folder(s) into {local_ts_dir}/")
                for entry in entries:
                    remote_spec = f'{user}@{hostname}:{_REMOTE_MODELS_DIR}/{entry}'
                    cmd = scp_base + ['-r', remote_spec, str(local_ts_dir) + '/']
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)

            print("Transfer complete.")
            return

    elif transfer_type == 'push':
        # Ensure remote destination directory exists
        mkdir_cmd = ssh_base + [f'mkdir -p {remote_path}']
        print(f"Ensuring remote directory exists: {remote_path}")
        subprocess.run(mkdir_cmd, check=True)
        if folder:
            # Collect only files (not subdirectories) from local folder
            local_dir = Path(local_path)
            files = sorted(str(f) for f in local_dir.iterdir() if f.is_file())
            if not files:
                print("No files found in local directory.")
                return
            remote_spec = f'{user}@{hostname}:{remote_path}'
            cmd = scp_base + files + [remote_spec]
        else:
            remote_spec = f'{user}@{hostname}:{remote_path}'
            cmd = scp_base + ['-r', local_path, remote_spec]
    else:
        raise ValueError(f"transfer_type must be 'pull' or 'push', got '{transfer_type}'")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Transfer complete.")


def main() -> None:
    '''Entry point for the SCP CLI.'''
    parser = argparse.ArgumentParser(
        description='SCP model files to/from a remote server.'
    )
    parser.add_argument(
        'server',
        help=(
            'Server to connect to. Either IP:port (e.g. 194.68.245.60:22050) '
            'or an SSH config host name (e.g. RunPodServer1).'
        ),
    )
    parser.add_argument(
        '--type',
        dest='transfer_type',
        choices=['pull', 'push'],
        default='pull',
        help="Transfer direction: 'pull' (download, default) or 'push' (upload).",
    )
    parser.add_argument(
        '--model-folder',
        default=None,
        help='Name of a specific model folder to transfer. If omitted, transfers all.',
    )

    args = parser.parse_args()
    scp_transfer(
        server=args.server,
        transfer_type=args.transfer_type,
        folder=args.model_folder,
    )


if __name__ == '__main__':
    main()
