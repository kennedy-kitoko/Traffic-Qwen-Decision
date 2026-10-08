#!/usr/bin/env python3
"""Verify repository and optional local-checkpoint hashes."""
import argparse, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def verify(path):
    errors=[]; checked=0
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        expected,rel=line.split(None,1); rel=rel.strip().lstrip("*"); item=ROOT/rel
        if not item.is_file(): errors.append(f"missing: {rel}")
        elif digest(item)!=expected: errors.append(f"hash mismatch: {rel}")
        else: checked+=1
    return checked,errors
def main():
    p=argparse.ArgumentParser(); p.add_argument("--include-local-checkpoint",action="store_true"); a=p.parse_args()
    n,errors=verify(ROOT/"artifacts_manifest/checksums.sha256")
    if a.include_local_checkpoint:
        c,e=verify(ROOT/"checkpoints/manifests/adapter.sha256"); n+=c; errors+=e
    print(f"verified={n}, errors={len(errors)}")
    for error in errors: print(error)
    if errors: sys.exit(1)
if __name__=="__main__": main()
