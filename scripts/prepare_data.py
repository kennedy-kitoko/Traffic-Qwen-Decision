#!/usr/bin/env python3
"""Rebuild two-choice V1 data from an authorized guarded state-action trace."""
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--trace",required=True,type=Path); p.add_argument("--output-root",type=Path,default=ROOT/"data/traffic_qwen"); a=p.parse_args()
    if not a.trace.is_file(): p.error(f"Trace does not exist: {a.trace}")
    raw=a.output_root/"v1"; dual=a.output_root/"v1_dual_choice"
    subprocess.run([sys.executable,str(ROOT/"traffic_qwen/build_imitation_dataset.py"),"--trace",str(a.trace),"--out",str(raw)],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/"traffic_qwen/build_dual_choice_dataset.py"),"--source",str(raw),"--output",str(dual)],check=True,cwd=ROOT)
    print(f"Wrote splits to {dual}; compare hashes/counts with data/manifests before training.")
if __name__=="__main__": main()
