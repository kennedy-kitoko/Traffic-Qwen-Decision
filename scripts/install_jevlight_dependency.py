#!/usr/bin/env python3
import pathlib, subprocess
root=pathlib.Path(__file__).resolve().parents[1]
subprocess.run(["bash",str(root/"scripts/install_jevlight_dependency.sh")],check=True)
