"""Download official Qwen weights from ModelScope and verify every SHA-256."""

from pathlib import Path as _RuntimePath
import sys as _runtime_sys
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
from runtime import ensure_project_python
ensure_project_python(__file__)

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'artifacts' / 'model_download'
DEST = ROOT / 'models' / 'Qwen3.5-0.8B'
API = 'https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-0.8B/repo/files'
DOWNLOAD = 'https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-0.8B/repo'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(meta):
    relative = Path(meta['Path'])
    target = (DEST / relative).resolve()
    if not target.is_relative_to(DEST.resolve()):
        raise ValueError('Unsafe repository path')
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = meta['Sha256']
    if target.exists() and target.stat().st_size == meta['Size'] and sha256(target) == expected:
        print(f'VERIFIED existing {relative}', flush=True)
        return
    partial = target.with_name(target.name + '.partial')
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset != meta['Size']:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with requests.get(DOWNLOAD, params={'Revision': meta['Revision'], 'FilePath': meta['Path']},
                                  headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0
                    elif offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected resume range')
                    with partial.open('ab' if offset else 'wb') as stream:
                        last = time.monotonic()
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            stream.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last >= 15:
                                print(f'DOWNLOAD {relative}: {offset / 1e6:.1f}/{meta["Size"] / 1e6:.1f} MB', flush=True)
                                last = time.monotonic()
            if partial.stat().st_size != meta['Size'] or sha256(partial) != expected:
                partial.unlink()
                raise RuntimeError(f'Checksum mismatch for {relative}')
            partial.replace(target)
            print(f'VERIFIED {relative} ({meta["Size"]} bytes)', flush=True)
            return
        except Exception as exc:
            print(f'RETRY {relative} {attempt + 1}: {exc}', flush=True)
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    response = requests.get(API, params={'Revision': 'master', 'Recursive': 'true'}, timeout=30)
    response.raise_for_status()
    listing = response.json()
    if listing.get('Code') != 200:
        raise RuntimeError(listing)
    files = [x for x in listing['Data']['Files'] if x['Type'] == 'blob' and x['Path'] != '.gitattributes']
    (REPORTS / 'modelscope_manifest.json').write_text(json.dumps(listing, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Downloading {len(files)} files, {sum(x["Size"] for x in files) / 1e9:.2f} GB', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, files))
    report = {'model': 'Qwen/Qwen3.5-0.8B', 'source': API, 'verified_sha256': True,
              'files': [{'path': x['Path'], 'bytes': x['Size'], 'sha256': x['Sha256']} for x in files]}
    (REPORTS / 'download_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE: all model files verified by SHA-256', flush=True)


if __name__ == '__main__':
    main()
