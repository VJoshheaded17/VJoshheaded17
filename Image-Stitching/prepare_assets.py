"""Restore bundled PNG bytes after cloning. Requires only the Python standard library."""
import base64
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    entries = json.loads((root / 'assets/manifest.json').read_text())
    restored = 0
    for entry in entries:
        source = (root / entry['encoded_path']).resolve()
        target = (root / entry['png_path']).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(root):
            raise ValueError('Asset path escapes the project')
        data = base64.b64decode(source.read_text(), validate=True)
        expected = entry['sha256']
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Asset checksum mismatch: {entry["png_path"]}')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                continue
            raise ValueError(f'Refusing to overwrite a modified image: {entry["png_path"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        restored += 1
    print(f'Verified {len(entries)} PNG assets; restored {restored}.')


if __name__ == '__main__':
    main()
