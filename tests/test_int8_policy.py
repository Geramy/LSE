#!/usr/bin/env python3
"""Compare real emitted kernel cache identities across independent policies."""
import os
import re
import subprocess
import sys

results = {}
for name, value, expected in [("unset", None, "0"), ("off", "0", "0"),
                              ("invalid", "true", "0"), ("on", "1", "1")]:
    env = os.environ.copy()
    env.pop("LSE_HRX_INT8", None)
    if value is not None:
        env["LSE_HRX_INT8"] = value
    env["LSE_WMMA_MIN_M"] = "16"
    env.pop("LSE_WMMA", None)
    completed = subprocess.run([sys.argv[1], expected], env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=90, check=False)
    print(f"{name}:\n{completed.stdout}", end="")
    completed.check_returncode()
    results[name] = dict((key, (hip, loom)) for key, hip, loom in
                        re.findall(r"KEY (\S+) hip=(\d+) loom=(\d+)", completed.stdout))
    assert len(results[name]) == 6
assert results["unset"] == results["off"] == results["invalid"]
for shape, keys in results["off"].items():
    for dialect in (0, 1):
        assert keys[dialect] != results["on"][shape][dialect], (shape, dialect)
print("PASS cross-process HIP/Loom cache separation and canonical disabled policy")
