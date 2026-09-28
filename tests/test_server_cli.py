#!/usr/bin/env python3
"""Validate CLI options without backend initialization, model loading, or sockets."""
import os
from pathlib import Path
import subprocess
import sys

if len(sys.argv) not in (2, 3):
    raise SystemExit('Usage: test_server_cli.py /path/to/lse-server [/path/to/lse]')
server = Path(sys.argv[1])
env = dict(os.environ)
env.pop('LSE_MODEL', None)
cases = [
    (['--help'], 0, '--dialect NAME'),
    (['--dialect', 'loom', '--help'], 0, 'Endpoints:'),
    (['--dialect', 'hip', '--help'], 0, 'Endpoints:'),
    (['--dialect', 'unknown', '--model', '/not-opened'], 2, "no dialect is spelled 'unknown'"),
    (['--dialect'], 2, '--dialect needs a value'),
    (['--dialect', 'loom'], 2, 'no model.'),
]
for value in ('1', '30', '600'):
    cases.append((['--shutdown-grace-seconds', value, '--help'], 0,
                  '--shutdown-grace-seconds'))
for value in ('0', '601', '-1', '1.5', '30x', '', '99999999999999999999'):
    cases.append((['--shutdown-grace-seconds', value], 2,
                  'shutdown grace must be an integer'))
cases.append((['--shutdown-grace-seconds'], 2,
              '--shutdown-grace-seconds needs a value'))
for value in ('1', '2', '7'):
    cases.append((['--mtp-depth', value, '--help'], 0, '--mtp-depth N'))
for value in ('0', '8', '-1', '2.0', '2x', '', '99999999999999999999'):
    cases.append((['--mtp-depth', value], 2, 'MTP depth must be an integer'))
cases.append((['--mtp-depth'], 2, '--mtp-depth needs a value'))
for value in ('on', 'off'):
    cases += [([f'--dflash2={value}', '--help'], 0, '--dflash2=on'),
              (['--dflash2', value, '--help'], 0, '--dflash2=on')]
for value in ('true', 'false', '1', '', 'ON'):
    cases.append(([f'--dflash2={value}'], 2, '--dflash2 must be on or off'))
cases.append((['--dflash2'], 2, '--dflash2 needs a value'))
cases.append((['--dflash2-model'], 2, '--dflash2-model needs a value'))
for args, code, message in cases:
    result = subprocess.run([str(server), *args], env=env, capture_output=True, text=True, timeout=10)
    output = result.stdout + result.stderr
    if result.returncode != code or message not in output:
        raise SystemExit(f'FAIL {args!r}: exit={result.returncode}\n{output}')
print(f'PASS {len(cases)} server CLI cases; no backend, model, or HTTP server opened')

if len(sys.argv) == 3:
    cli = Path(sys.argv[2])
    cli_cases = [(['--mtp-depth', value, '--help'], 0, '--mtp-depth N')
                 for value in ('1', '2', '7')]
    cli_cases += [(['--mtp-depth', value], 2, 'MTP depth must be an integer')
                  for value in ('0', '8', '-1', '2.0', '2x', '', '99999999999999999999')]
    cli_cases.append((['--mtp-depth'], 2, '--mtp-depth needs a value'))
    for value in ('on', 'off'):
        cli_cases += [([f'--dflash2={value}', '--help'], 0, '--dflash2=on'),
                      (['--dflash2', value, '--help'], 0, '--dflash2=on')]
    for value in ('true', 'false', '1', '', 'ON'):
        cli_cases.append(([f'--dflash2={value}'], 2, '--dflash2 must be on or off'))
    cli_cases.append((['--dflash2'], 2, '--dflash2 needs a value'))
    cli_cases.append((['--dflash2-model'], 2, '--dflash2-model needs a value'))
    for args, code, message in cli_cases:
        result = subprocess.run([str(cli), *args], env=env, capture_output=True,
                                text=True, timeout=10)
        output = result.stdout + result.stderr
        if result.returncode != code or message not in output:
            raise SystemExit(f'FAIL CLI {args!r}: exit={result.returncode}\n{output}')
    print(f'PASS {len(cli_cases)} CLI MTP depth cases; no backend opened')
