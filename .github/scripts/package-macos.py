#!/usr/bin/env python3
"""Bundle the native runtime closure and verify it after relocation, without a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


def run(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs)


def dependencies(path):
    return [line.strip().split(' (compatibility')[0]
            for line in run('otool', '-L', str(path)).splitlines()[1:]]


def rpaths(path):
    lines = run('otool', '-l', str(path)).splitlines()
    return [lines[i + 2].strip().split(' ', 2)[1]
            for i, line in enumerate(lines) if line.strip() == 'cmd LC_RPATH']


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'work', 'llvm', 'tag'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, work, llvm = (Path(x).resolve() for x in (args.root, args.work, args.llvm))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag):
        raise SystemExit('Invalid archive tag')
    name = f'lse-{args.tag}-macos-arm64'
    dist = root / 'dist'
    package = dist / name
    if package.exists():
        shutil.rmtree(package)
    for directory in ('bin', 'lib', 'libexec', 'licenses'):
        (package / directory).mkdir(parents=True)
    search = [llvm / 'lib/c++', llvm / 'lib/unwind', llvm / 'lib',
              work / 'hrx-build/loom/binding/c', work / 'hrx-build/libhrx/src/libhrx',
              work / 'hsa-build']
    sources = {}
    pending = []

    def add(src, dst):
        src = src.resolve()
        if dst in sources:
            if sources[dst] != src:
                raise RuntimeError(f'Conflicting dependency basename: {src} / {sources[dst]}')
            return
        sources[dst] = src
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | 0o200)
        pending.append(dst)

    for binary in ('lse', 'lse-server', 'compile_loom_matrix'):
        add(work / 'lse-build' / ('tests/' + binary if binary == 'compile_loom_matrix' else binary),
            package / 'libexec' / binary)
    # HRX opens HSA by basename at runtime; it is absent from otool's closure.
    add(work / 'hsa-build/libhsa-runtime64.dylib', package / 'lib/libhsa-runtime64.1.dylib')
    (package / 'lib/libhsa-runtime64.dylib').symlink_to('libhsa-runtime64.1.dylib')
    for dst in pending:
        src = sources[dst]
        own_id = run('otool', '-D', str(src)).splitlines()[1:]
        for dep in dependencies(src):
            if dep in own_id or dep.startswith(('/System/', '/usr/lib/')):
                continue
            if dep.startswith('@rpath/'):
                suffix = dep[len('@rpath/'):]
                bases = [Path(p.replace('@loader_path', str(src.parent))) for p in rpaths(src)]
                candidates = [p / suffix for p in bases + search]
            elif dep.startswith('@loader_path/'):
                candidates = [src.parent / dep[len('@loader_path/'):]]
            elif dep.startswith('/'):
                candidates = [Path(dep)]
                # A retained Homebrew keg may have an old unversioned install
                # name; prefer the explicitly qualified LLVM 21 runtime.
                if '/llvm/' in dep or '/llvm@21/' in dep:
                    candidates = [p / Path(dep).name for p in search] + candidates
            else:
                raise RuntimeError(f'Unsupported dependency: {src}: {dep}')
            resolved = next((p for p in candidates if p.is_file()), None)
            if resolved is None:
                raise RuntimeError(f'Unresolved dependency: {src}: {dep}')
            target = package / 'lib' / Path(dep).name
            add(resolved, target)
            subprocess.run(['install_name_tool', '-change', dep, '@rpath/' + target.name, str(dst)], check=True)
        for old in rpaths(dst):
            subprocess.run(['install_name_tool', '-delete_rpath', old, str(dst)], check=True)
        relative = '@loader_path' if dst.parent.name == 'lib' else '@loader_path/../lib'
        subprocess.run(['install_name_tool', '-add_rpath', relative, str(dst)], check=True)
        if dst.suffix == '.dylib':
            subprocess.run(['install_name_tool', '-id', '@rpath/' + dst.name, str(dst)], check=True)
        subprocess.run(['codesign', '--force', '--sign', '-', str(dst)], check=True)
        for dep in dependencies(dst):
            if not dep.startswith(('/System/', '/usr/lib/', '@rpath/')):
                raise RuntimeError(f'Nonportable dependency in package: {dst}: {dep}')
    # The original compiler library path is unavailable after relocation.
    # Isolate the default JIT cache by immutable source/toolchain identity.
    build_inputs = {
        'lse': run('git', '-C', str(root), 'rev-parse', 'HEAD').strip(),
        'portability_patch': hashlib.sha256((root / '.github/patches/macos-portability.patch').read_bytes()).hexdigest(),
        'hrx': run('git', '-C', str(work / 'deps/hrx'), 'rev-parse', 'HEAD').strip(),
        'mac_amdgpu': run('git', '-C', str(work / 'deps/mac-amdgpu'), 'rev-parse', 'HEAD').strip(),
        'llvm': run(str(llvm / 'bin/llvm-config'), '--version').strip(),
    }
    cache_identity = hashlib.sha256(json.dumps(build_inputs, sort_keys=True).encode()).hexdigest()[:24]
    for binary in ('lse', 'lse-server'):
        wrapper = package / 'bin' / binary
        wrapper.write_text('#!/bin/sh\nset -eu\n'
            'package_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"\n'
            'export DYLD_LIBRARY_PATH="$package_root/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"\n'
            'export LSE_CACHE_DIR="${LSE_CACHE_DIR:-$HOME/Library/Caches/lse/' + cache_identity + '}"\n'
            f'exec "$package_root/libexec/{binary}" "$@"\n')
        wrapper.chmod(0o755)
    for src, dst in ((root / 'LICENSE.md', 'LSE.md'),
                     (work / 'deps/mac-amdgpu/LICENSE', 'MacAMDGPU'),
                     (work / 'hrx-source/LICENSE', 'HRX.txt'),
                     (llvm / 'LICENSE.TXT', 'LLVM.txt')):
        if not src.is_file():
            alternatives = [src.with_suffix('.md'), src.with_suffix('.txt')]
            src = next((p for p in alternatives if p.is_file()), src)
        if not src.is_file():
            raise RuntimeError(f'Missing dependency license: {src}')
        shutil.copy2(src, package / 'licenses' / dst)
    # Include the upstream notices for statically linked third-party code.
    for depdir, prefix in ((work / 'hrx-source', 'hrx'), (work / 'hrx-build/_deps', 'hrx-deps'),
                            (work / 'source/third_party/vendor', 'lse-vendor'),
                            (work / 'source/reference/fastokens', 'fastokens')):
        for license_file in depdir.rglob('*'):
            if license_file.is_file() and license_file.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')):
                target = package / 'licenses' / prefix / license_file.relative_to(depdir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(license_file, target)
    crate = work / 'source/third_party/fastokens-ffi/Cargo.toml'
    metadata = json.loads(run('cargo', 'metadata', '--locked', '--offline',
                              '--filter-platform', 'aarch64-apple-darwin',
                              '--format-version', '1', '--manifest-path', str(crate)))
    rust_notices = []
    for dep in metadata['packages']:
        directory = Path(dep['manifest_path']).parent
        notice = {'name': dep['name'], 'version': dep['version'], 'license': dep['license']}
        rust_notices.append(notice)
        for license_file in directory.iterdir():
            if license_file.is_file() and license_file.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')):
                target = package / 'licenses/rust' / f"{dep['name']}-{dep['version']}" / license_file.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(license_file, target)
    (package / 'licenses/rust-dependencies.json').write_text(json.dumps(rust_notices, indent=2) + '\n')
    shutil.copy2(root / 'README.md', package / 'README.md')
    manifest = {
        'lse_revision': run('git', '-C', str(root), 'rev-parse', 'HEAD').strip(),
        'mac_amdgpu_revision': run('git', '-C', str(work / 'deps/mac-amdgpu'), 'rev-parse', 'HEAD').strip(),
        'hrx_revision': run('git', '-C', str(work / 'deps/hrx'), 'rev-parse', 'HEAD').strip(),
        'llvm': run(str(llvm / 'bin/llvm-config'), '--version').strip(),
        'cargo': run('cargo', '--version').strip(),
        'host': run('sw_vers').strip(),
        'architecture': run('uname', '-m').strip(),
        'gpu_target': 'gfx1201',
        'dialect': 'loom',
        'gpu_execution_tested_in_ci': False,
        'jit_cache_identity': cache_identity,
        'jit_cache_build_inputs': build_inputs,
        'portability_patch_sha256': hashlib.sha256((root / '.github/patches/macos-portability.patch').read_bytes()).hexdigest(),
    }
    (package / 'BUILD.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (package / 'QUICKSTART.txt').write_text(
        'Binary deployment target: Apple Silicon macOS 15 or newer.\n'
        'External AMD GPU use requires the driver-supported macOS version\n'
        '(currently macOS Tahoe 26.2+); gfx1201/R9700 qualification target.\n'
        'Install and enable the MacAMDGPU DriverKit extension separately:\n'
        'https://github.com/lemonade-sdk/mac-amdgpu\n'
        'The extension is not bundled or installed by this archive.\n\n'
        './bin/lse --help\n'
        './bin/lse --model /path/to/model --pool hrx:0 --dialect loom --prompt "Hello"\n'
        './bin/lse-server --model /path/to/model --pool hrx:0 --dialect loom\n\n'
        'Use the bin wrappers so HRX finds the bundled HSA runtime.\n'
        'Default JIT cache: ~/Library/Caches/lse/<build identity>.\n'
        'This prevents stale kernel reuse between relocated release packages.\n'
        'An explicit LSE_CACHE_DIR overrides that default.\n'
        'HIPC/comgr is unavailable in this macOS package.\n'
        'CI checks host behavior, relocation, and native kernel compilation;\n'
        'CI runners do not have an external AMD GPU and do not execute GPU kernels.\n'
        'Binaries are ad-hoc signed, not Developer ID notarized.\n')
    # An unrelated path with spaces catches embedded build paths and quoting.
    # Empty library-path variables stop the developer machine masking omissions.
    with tempfile.TemporaryDirectory(prefix='lse relocated package ') as tmp:
        relocated = Path(tmp) / name
        shutil.copytree(package, relocated, symlinks=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith('DYLD_')}
        for binary in ('lse', 'lse-server'):
            subprocess.run([str(relocated / 'bin' / binary), '--help'], env=env,
                           check=True, stdout=subprocess.DEVNULL)
        # Compile all six shared matrix operand variants using the relocated
        # compiler library; this opens no HSA runtime and requires no hardware.
        output = Path(tmp) / 'matrix'
        output.mkdir()
        subprocess.run([str(relocated / 'libexec/compile_loom_matrix'), str(output)], env=env, check=True)
        # Load the late-bound runtime without hsa_init / opening a device.
        subprocess.run(['/usr/bin/python3', '-c',
                        'import ctypes,sys; ctypes.CDLL(sys.argv[1])',
                        str(relocated / 'lib/libhsa-runtime64.1.dylib')], env=env, check=True)
    # The compiler fixture is a build check, not a public inference command.
    (package / 'libexec/compile_loom_matrix').unlink()
    archive = dist / (name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(package, arcname=name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.gz.sha256').write_text(f'{digest}  {archive.name}\n')
    print(archive)


if __name__ == '__main__':
    main()
