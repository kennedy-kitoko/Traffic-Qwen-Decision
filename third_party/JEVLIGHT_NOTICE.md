# JevLight notice

Upstream: https://github.com/usail-hkust/JevLight
License: MIT; full upstream notice is preserved in JEVLIGHT_LICENSE.
Pinned source commit: d92caa655e5a7487a49361626673df809eee9129.

The measured JevLight worktree had local changes to utils/config.py and utils/oneline.py to register and run TrafficQwen. The diff is preserved as jevlight-runtime.patch; the installer verifies it applies to the pinned commit. Traffic-Qwen agent code is included separately. No JevLight files in the original repository are changed by this staging operation.
