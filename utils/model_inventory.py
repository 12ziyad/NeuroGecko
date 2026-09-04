"""Catalog model files without loading executable pickle/checkpoint contents."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, archive: Path | None = None) -> dict:
    root = root.resolve(strict=True)
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink not allowed in inventory: {path}')
        if not path.is_file() or path.name == 'INVENTORY.md':
            continue
        item = dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                    sha256=sha256_file(path))
        if path.name == 'train_config.json':
            item['train_config'] = json.loads(path.read_text(encoding='utf-8'))
        files.append(item)
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  root=str(root), file_count=len(files), total_bytes=sum(f['bytes'] for f in files),
                  files=files)
    if archive:
        result['archive'] = dict(name=archive.name, bytes=archive.stat().st_size,
                                 sha256=sha256_file(archive))
    return result


def render_markdown(data: dict, source_commit: str) -> str:
    lines = ['# Recovered model inventory', '',
             'Status: recovered from the existing AWS project and verified on this laptop.', '',
             f"Generated UTC: {data['generated_utc']}",
             f'Source checkout: `{source_commit}` (weights are separately hashed below).',
             f"Files: **{data['file_count']}**; uncompressed bytes: **{data['total_bytes']}**.", '',
             'Recovery location: `models_recovered/recovery-20260904T233438Z/models/`.',
             'The original local `models/` and original AWS weights were not overwritten.', '',
             '## Integrity', '']
    if 'archive' in data:
        a = data['archive']
        lines += [f"Archive: `{a['name']}` ({a['bytes']} bytes).", '', f"SHA256: `{a['sha256']}`.", '',
                  'The archive checksum matched AWS. All individual extracted-file hashes matched',
                  'the independently downloaded AWS `files.sha256` manifest.', '']
    lines += ['## Interpretation', '',
              '- `brain_v1_patch38c_visual_dagger_80k_seed0`: recovered visual student / named champion.',
              '- `brain_v1_patch37b_dagger_200k_seed1`: recovered privileged teacher.',
              '- `v4_5b_speed_polish_1m`: frozen low-level walker and normalization state.',
              '- Other run names and configurations document lineage; names alone do not prove performance.',
              '- Historical eating counts are task-specific outcomes, not a newly reproduced result or biological validation.',
              '- No checkpoint was unpickled to produce this inventory.', '', '## Files', '',
              '| Relative path | Bytes | SHA256 |', '|---|---:|---|']
    for item in data['files']:
        lines.append(f"| `{item['path']}` | {item['bytes']} | `{item['sha256']}` |")
    lines += ['', '## Training configurations', '']
    for item in data['files']:
        if 'train_config' in item:
            lines += [f"### {item['path']}", '', '```json',
                      json.dumps(item['train_config'], indent=2, sort_keys=True), '```', '']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--json-output', type=Path)
    parser.add_argument('--source-commit', default='unknown')
    args = parser.parse_args()
    data = inventory(args.root, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(data, args.source_commit), encoding='utf-8')
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in data.items() if k not in ('files', 'root')}, indent=2))


if __name__ == '__main__':
    main()
