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
    model_folder: str | None = None,
) -> None:
    '''
    SCP files to or from a remote server.

    Args:
        server: Either an IP:port string (e.g. "194.68.245.60:22050") or an
                SSH config host name (e.g. "RunPodServer1").
        transfer_type: "pull" to download from remote, "push" to upload.
        model_folder: Optional name of a specific model folder. If omitted,
                      the entire models directory is transferred.
    '''
    hostname, port, user, identity_file = _resolve_server(server)

    if model_folder:
        remote_path = f'{_REMOTE_MODELS_DIR}/{model_folder}/'
        local_path = str(_LOCAL_MODELS_DIR / model_folder)
    else:
        remote_path = f'{_REMOTE_MODELS_DIR}/'
        local_path = str(_LOCAL_MODELS_DIR) + '/'

    remote_spec = f'{user}@{hostname}:{remote_path}'

    if transfer_type == 'pull':
        # Ensure local destination exists
        os.makedirs(local_path, exist_ok=True)
        src, dst = remote_spec, local_path
    elif transfer_type == 'push':
        src, dst = local_path, remote_spec
    else:
        raise ValueError(f"transfer_type must be 'pull' or 'push', got '{transfer_type}'")

    cmd = ['scp', '-r', '-P', port, '-i', identity_file, src, dst]
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
        '--model_folder',
        default=None,
        help='Name of a specific model folder to transfer. If omitted, transfers all.',
    )

    args = parser.parse_args()
    scp_transfer(
        server=args.server,
        transfer_type=args.transfer_type,
        model_folder=args.model_folder,
    )


if __name__ == '__main__':
    main()
