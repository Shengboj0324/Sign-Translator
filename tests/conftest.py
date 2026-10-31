"""Suite-wide guard: tests never see real storage or cloud credentials.

Launcher tests run the real fleet scripts. On a pod, where credentials are in the environment, one of them ran a real
smoke job against R2 and outlived its timeout (2026-10-31).
"""

import os

for _name in list(os.environ):
    if _name.startswith(("R2_", "AWS_", "RUNPOD_")):
        del os.environ[_name]
