#!/usr/bin/env python3
"""Validate CLI options without backend initialization, model loading, or sockets."""
import json
import tempfile
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
    (['--help'], 0, 'source dialect: loom or hip (default: loom)'),
    (['--dialect', 'loom', '--help'], 0, 'Endpoints:'),
    (['--dialect', 'hip', '--help'], 0, 'Endpoints:'),
    (['--dialect', 'unknown', '--model', '/not-opened'], 2, "no dialect is spelled 'unknown'"),
    (['--dialect'], 2, '--dialect needs a value'),
    (['--dialect', 'loom'], 2, 'no model.'),
    (['--help'], 0, '--no-cpu-fallback'),
    (['--no-cpu-fallback'], 2, 'no model.'),
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
cases.append((['--help'], 0, 'default: bf16 for BF16 models, fp16 otherwise'))
cases.append((['--help'], 0, '--perplexity FILE'))
for flag in ('--perplexity-ctx', '--perplexity-stride', '--perplexity-kld-top-k'):
    cases.append(([flag, '16', '--help'], 0, '--perplexity-ctx N'))
    for value in ('0', '-1', '1.5', '64x', ''):
        cases.append(([flag, value], 2, flag + ' must be a positive integer'))
    cases.append(([flag], 2, flag + ' needs a value'))
cases += [(['--perplexity-kld-top-k', '33'], 2, 'up to 32'),
          (['--perplexity-method', 'llama'], 2, 'must be chunks or sliding'),
          (['--perplexity-method', 'sliding', '--help'], 0, '--perplexity-method M'),
          (['--perplexity-chunks', '0', '--help'], 0, '--perplexity-chunks N'),
          (['--perplexity-chunks', 'x'], 2, 'must be a positive integer'),
          (['--perplexity', '/not-a-file', '--model', '/not-opened'], 2, 'cannot read /not-a-file'),
          (['--perplexity', '/not-a-file', '--perplexity-tokens', '/not-a-file', '--model', '/not-opened'],
           2, 'not both')]
for value in ('fp32', 'fp16', 'bf16', 'fp8', 'bf8'):
    cases.append((['--kv-cache-dtype', value, '--help'], 0, '--kv-cache-dtype TYPE'))
for value in ('int8', '', 'invalid'):
    cases.append((['--kv-cache-dtype', value], 2, 'KV cache dtype must be'))
cases.append((['--kv-cache-dtype'], 2, '--kv-cache-dtype needs a value'))
for flag in ('--batch-size', '--ubatch-size'):
    for value in ('128', '512', '1024', '2048', '4096'):
        cases.append((['--batch-size', '4096', flag, value, '--help'], 0, '--ubatch-size N'))
    for value in ('0', '-1', '1023', '8192', '1.5', '1024x', '', '99999999999999999999'):
        cases.append(([flag, value], 2, 'prefill batch size must be a power of two'))
    cases.append(([flag], 2, flag + ' needs a value'))
cases.append((['--batch-size', '1024', '--ubatch-size', '2048', '--model', 'unused'],
              2, '--ubatch-size must not exceed --batch-size'))
for value in ('0', '0.6', '1', '2'):
    cases.append((['--temperature', value, '--help'], 0, '--temperature F'))
for value in ('-1', '2.1', 'nan', 'inf', '0.6x', ''):
    cases.append((['--temperature', value], 2, 'temperature must be a finite number'))
cases.append((['--temperature'], 2, '--temperature needs a value'))
for value in ('on', 'off'):
    cases += [([f'--FlashPrefillV2={value}', '--help'], 0, '--FlashPrefillV2=off'),
              (['--FlashPrefillV2', value, '--help'], 0, '--FlashPrefillV2=off')]
for value in ('true', 'false', '1', '', 'ON'):
    cases.append(([f'--FlashPrefillV2={value}'], 2, '--FlashPrefillV2 must be on or off'))
cases += [(['--FlashPrefillV2'], 2, '--FlashPrefillV2 needs a value'),
          (['--FlashPrefillV2=off'], 2, 'no model.'),
          # Loom is the default dialect, so the experimental attention modes
          # that need it are accepted without naming it, and refused only
          # when the legacy HIP dialect is named.
          (['--FlashPrefillV2=on'], 2, 'no model.'),
          (['--FlashPrefillV2=on', '--dialect', 'hip'], 2, 'requires --dialect loom'),
          (['--FlashPrefillV2=on', '--no-mtp'], 2, 'no model.'),
          (['--FlashPrefillV2=on', '--no-mtp', '--dialect', 'loom'], 2, 'no model.'),
          (['--attention-prefill', 'flashprefill-v2', '--no-mtp', '--dialect', 'loom'], 2, 'no model.'),
          (['--FlashPrefillV2=off', '--dflash2=on'], 2, 'no model.'),
          (['--FlashPrefillV2=on', '--mtp-depth', '3', '--dialect', 'loom'], 2, 'no model.'),
          (['--FlashPrefillV2=on', '--dflash2=on', '--dialect', 'loom'], 2, 'no model.'),
          (['--attention-prefill', 'blasst', '--dflash2=on', '--dialect', 'loom'], 2, 'requires --no-mtp')]
for flags in (['--FlashPrefillV2=off', '--attention-prefill', 'flashprefill-v2'],
              ['--attention-prefill', 'flashprefill-v2', '--FlashPrefillV2=off'],
              ['--FlashPrefillV2=on', '--attention-prefill', 'dense'],
              ['--attention-prefill', 'dense', '--FlashPrefillV2=on']):
    cases.append((flags, 2, 'conflicting --FlashPrefillV2'))
for flag in ('--attention-prefill', '--attention-decode'):
    for mode in ('dense', 'blasst'):
        cases.append(([flag, mode, '--help'], 0, '--attention-calibration'))
    for mode in ('unknown', ''):
        cases.append(([flag, mode], 2, 'attention mode must be dense or blasst'))
    cases.append(([flag], 2, flag + ' needs a value'))
cases += [(['--attention-prefill', 'flashprefill-v2', '--help'], 0, '--attention-calibration'),
          (['--attention-decode', 'flashprefill-v2'], 2, 'prefill only'),
          (['--attention-prefill', 'blasst'], 2, 'requires --no-mtp'),
          (['--attention-decode', 'blasst', '--no-mtp'], 2, 'requires --attention-calibration'),
          (['--attention-calibration'], 2, '--attention-calibration needs a value'),
          (['--attention-decode', 'blasst', '--no-mtp', '--attention-calibration', '/not-present'], 2, 'invalid attention calibration'),
          (['--attention-decode', 'blasst', '--no-mtp', '--attention-calibration', '/not-present', '--dialect', 'hip'], 2, 'requires --dialect loom'),
          (['--attention-decode', 'blasst', '--no-mtp', '--attention-calibration', '/not-present', '--dialect', 'loom'], 2, 'invalid attention calibration')]
calibration_dir = tempfile.TemporaryDirectory()
for index, (payload, mode, message) in enumerate([
    ({'version': 2, 'model': '/not-opened'}, 'blasst', 'unsupported calibration version'),
    ({'version': 1, 'model': '/wrong'}, 'blasst', 'match --model exactly'),
    ({'version': 1, 'model': '/not-opened', 'prefill': {'scale': -1}}, 'blasst', 'finite, nonnegative'),
    ({'version': 1, 'model': '/not-opened', 'prefill': {'scale': '0'}}, 'blasst', 'must be numeric'),
    ({'version': 1, 'model': '/not-opened', 'prefill': {'scale': 1.1}}, 'flashprefill-v2', 'at most 1'),
]):
    calibration = Path(calibration_dir.name) / f'{index}.json'
    calibration.write_text(json.dumps(payload))
    cases.append((['--model', '/not-opened', '--no-mtp', '--dialect', 'loom',
                   '--attention-prefill', mode, '--attention-calibration', str(calibration)], 2, message))
for args, code, message in cases:
    result = subprocess.run([str(server), *args], env=env, capture_output=True, text=True, timeout=10)
    output = result.stdout + result.stderr
    if result.returncode != code or message not in output:
        raise SystemExit(f'FAIL {args!r}: exit={result.returncode}\n{output}')
calibration_dir.cleanup()
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
    cli_cases.append((['--help'], 0, 'default: bf16 for BF16 models, fp16 otherwise'))
    for value in ('fp32', 'fp16', 'bf16', 'fp8', 'bf8'):
        cli_cases.append((['--kv-cache-dtype', value, '--help'], 0, '--kv-cache-dtype TYPE'))
    for value in ('int8', '', 'invalid'):
        cli_cases.append((['--kv-cache-dtype', value], 2, 'KV cache dtype must be'))
    cli_cases.append((['--kv-cache-dtype'], 2, '--kv-cache-dtype needs a value'))
    for flag in ('--batch-size', '--ubatch-size'):
        for value in ('1024', '2048', '4096'):
            cli_cases.append((['--batch-size', '4096', flag, value, '--help'], 0, '--ubatch-size N'))
        for value in ('0', '-1', '1023', '8192', '1.5', '1024x', '', '99999999999999999999'):
            cli_cases.append(([flag, value], 2, 'prefill batch size must be a power of two'))
        cli_cases.append(([flag], 2, flag + ' needs a value'))
    cli_cases.append((['--batch-size', '1024', '--ubatch-size', '2048'],
                      2, '--ubatch-size must not exceed --batch-size'))
    cli_cases += [(['--help'], 0, '--no-cpu-fallback'),
                  (['--no-cpu-fallback'], 2, 'no model given')]
    for args, code, message in cli_cases:
        result = subprocess.run([str(cli), *args], env=env, capture_output=True,
                                text=True, timeout=10)
        output = result.stdout + result.stderr
        if result.returncode != code or message not in output:
            raise SystemExit(f'FAIL CLI {args!r}: exit={result.returncode}\n{output}')
    print(f'PASS {len(cli_cases)} CLI MTP depth cases; no backend opened')
