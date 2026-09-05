"""Pull immutable checkpoint bundles to a second machine, verify, then acknowledge.

Run this on the laptop, not on the source instance. Only strict known-host SSH is
used. Private keys stay on the laptop. An interrupted copy remains .incoming-*;
it is never published as a verified bundle. This is not an AWS billing watchdog.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
import uuid

EXPECTED_FILES = {'model.zip', 'vecnormalize.pkl', 'train_config.json'}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_manifest(manifest: dict) -> None:
    if manifest.get('schema_version') != 1 or manifest.get('complete') is not True:
        raise ValueError('Only complete version-1 checkpoint bundles may be synced')
    if type(manifest.get('num_timesteps')) is not int or manifest['num_timesteps'] < 0:
        raise ValueError('Invalid checkpoint timestep')
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != EXPECTED_FILES:
        raise ValueError('Bundle must contain exactly the expected model, normalizer and config')
    for name, item in files.items():
        if type(item.get('bytes')) is not int or item['bytes'] < 0:
            raise ValueError(f'Invalid size: {name}')
        if not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', '')):
            raise ValueError(f'Invalid SHA256: {name}')


def verify_bundle(path: Path, manifest: dict) -> None:
    validate_manifest(manifest)
    for name, expected in manifest['files'].items():
        file = path / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f'Missing regular checkpoint file: {file}')
        if file.stat().st_size != expected['bytes'] or file_hash(file) != expected['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {file}')


class Puller:
    def __init__(self, host: str, identity: Path, remote_root: str, local_root: Path):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+', host):
            raise ValueError('Use a plain user@hostname SSH target')
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', remote_root) or '..' in PurePosixPath(remote_root).parts:
            raise ValueError('Remote path must be absolute, without traversal or shell characters')
        self.host = host
        self.remote_root = PurePosixPath(remote_root)
        if local_root.is_symlink():
            raise ValueError('Local destination must not be a symlink')
        self.local_root = local_root.resolve()
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.options = ['-i', str(identity.resolve(strict=True)), '-o', 'BatchMode=yes',
                        '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15']

    def ssh(self, command: str) -> str:
        return subprocess.run(['ssh', *self.options, self.host, command], check=True,
                              capture_output=True, text=True, timeout=60).stdout

    def scp(self, source: str, destination: str) -> None:
        subprocess.run(['scp', *self.options, source, destination], check=True,
                       capture_output=True, text=True, timeout=180)

    def manifests(self) -> list[PurePosixPath]:
        quoted = shlex.quote(str(self.remote_root))
        pattern = shlex.quote(str(self.remote_root / 'step-*' / 'manifest.json'))
        command = f'if test -d {quoted}; then find {quoted} -mindepth 2 -maxdepth 2 -type f -path {pattern}; fi'
        paths = []
        for line in self.ssh(command).splitlines():
            path = PurePosixPath(line)
            rel = path.relative_to(self.remote_root)
            if len(rel.parts) == 2 and rel.parts[0].startswith('.step-') and '.incomplete-' in rel.parts[0]:
                continue
            if len(rel.parts) != 2 or not re.fullmatch(r'step-[0-9]+(?:-final)?', rel.parts[0]) or rel.parts[1] != 'manifest.json':
                raise ValueError('Unexpected checkpoint discovery path')
            paths.append(path)
        return sorted(paths)

    def sync_one(self, source: PurePosixPath) -> dict:
        bundle_name = source.parent.name
        destination = self.local_root / bundle_name
        incoming = self.local_root / ('.incoming-' + uuid.uuid4().hex)
        incoming.mkdir()
        self.scp(f'{self.host}:{source}', str(incoming / 'manifest.json'))
        manifest = json.loads((incoming / 'manifest.json').read_text(encoding='utf-8'))
        validate_manifest(manifest)
        manifest_hash = file_hash(incoming / 'manifest.json')
        reused = destination.exists()
        if reused:
            if destination.is_symlink() or file_hash(destination / 'manifest.json') != manifest_hash:
                raise ValueError(f'Refusing to overwrite a different existing bundle: {destination}')
            verify_bundle(destination, manifest)
            # This directory contains only the manifest just created by this call.
            (incoming / 'manifest.json').unlink()
            incoming.rmdir()
        else:
            for name in EXPECTED_FILES:
                self.scp(f'{self.host}:{source.parent / name}', str(incoming / name))
            verify_bundle(incoming, manifest)
            incoming.rename(destination)
        receipt = dict(verified=True, manifest_sha256=manifest_hash,
                       destination=str(destination), verified_utc=datetime.now(timezone.utc).isoformat())
        local_receipt = destination / 'backup_receipt.json'
        temporary_receipt = destination / ('.receipt-' + uuid.uuid4().hex)
        temporary_receipt.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary_receipt, local_receipt)
        remote_temporary = source.parent / ('.receipt-' + uuid.uuid4().hex)
        remote_receipt = source.parent / 'backup_receipt.json'
        self.scp(str(local_receipt), f'{self.host}:{remote_temporary}')
        self.ssh(f'mv -- {shlex.quote(str(remote_temporary))} {shlex.quote(str(remote_receipt))}')
        return dict(bundle=bundle_name, timestep=manifest['num_timesteps'], verified=True,
                    reused=reused, manifest_sha256=manifest_hash)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--local-root', type=Path, required=True)
    p.add_argument('--watch-seconds', type=float, default=0, help='0: one pass; otherwise finite polling period')
    p.add_argument('--poll-seconds', type=float, default=10)
    args = p.parse_args()
    if (not math.isfinite(args.watch_seconds) or not math.isfinite(args.poll_seconds)
            or args.watch_seconds < 0 or args.poll_seconds <= 0):
        p.error('Watch duration must be nonnegative; polling interval positive')
    puller = Puller(args.host, args.identity, args.remote_root, args.local_root)
    deadline = time.monotonic() + args.watch_seconds
    acknowledged = set()
    while True:
        for source in puller.manifests():
            if str(source) not in acknowledged:
                print(json.dumps(puller.sync_one(source)), flush=True)
                acknowledged.add(str(source))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.poll_seconds, remaining))


if __name__ == '__main__':
    main()
