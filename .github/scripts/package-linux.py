#!/usr/bin/env python3
"""Select the built Loom compiler and verify the Linux runtime after relocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(*args, env=None):
    return subprocess.check_output(args, text=True, env=env, stderr=subprocess.STDOUT)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmake_cache(build):
    values = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if match := re.match(r'([^/#:][^:]*):[^=]+=(.*)', line):
            values[match[1]] = match[2]
    return values


def dependencies(path, env):
    text = run('ldd', str(path), env=env)
    if 'not found' in text:
        raise RuntimeError(f'Unresolved runtime dependency: {path}\n{text}')
    result = {}
    for line in text.splitlines():
        if match := re.match(r'\s*(\S+)\s+=>\s+(/.*?)\s+\(0x[0-9a-fA-F]+\)', line):
            result[match[1]] = Path(match[2]).resolve(strict=True)
    return result


def soname(path):
    text = run('readelf', '-d', str(path))
    match = re.search(r'\(SONAME\).*\[([^\]]+)\]', text)
    if not match or '/' in match[1]:
        raise RuntimeError(f'Expected a portable library SONAME: {path}')
    return match[1]


def runtime_env(paths, inherit=True):
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    suffix = env.get('LD_LIBRARY_PATH', '') if inherit else ''
    env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in paths if p) + (':' + suffix if suffix else '')
    return env


def verify(binaries, library, env):
    selected = soname(library)
    closure = {}
    for binary in binaries:
        links = dependencies(binary, env)
        if links.get(selected) != library.resolve():
            raise RuntimeError(f'{binary} loads a different Loom compiler: {links.get(selected)}')
        run(str(binary), '--help', env=env)
        closure[binary.name] = {name: str(path) for name, path in links.items()}
    return closure


# Component licenses of TheRock's rocm_sysdeps libraries, which the tree does
# not ship as files. The libraries are dynamically linked and replaceable.
SYSDEPS_LICENSES = {
    'drm': 'libdrm: MIT',
    'drm_amdgpu': 'libdrm_amdgpu: MIT',
    'numa': 'libnuma (numactl): LGPL-2.1',
    'elf': 'libelf (elfutils): LGPL-3.0-or-later or GPL-2.0-or-later',
    'z': 'zlib: Zlib',
    'zstd': 'zstd: BSD-3-Clause',
    'liblzma': 'liblzma (xz): 0BSD',
    'bz2': 'libbz2 (bzip2): bzip2-1.0.6',
}


def bundle_hsa_runtime(root, package):
    """Copy an HSA runtime with hsa_amd_queue_create and its ROCm-owned closure.

    The runtime finds its dependencies through RUNPATH $ORIGIN and
    $ORIGIN/rocm_sysdeps/lib, so it keeps that layout under lib/ and its
    RUNPATH is left in place.
    """
    lib = root / 'lib'
    hsa = (lib / 'libhsa-runtime64.so.1').resolve(strict=True)
    if 'hsa_amd_queue_create' not in run('readelf', '--dyn-syms', '--wide', str(hsa)):
        raise RuntimeError(f'{hsa} lacks hsa_amd_queue_create; HRX needs ROCr 1.21 or newer')
    clean = runtime_env([], inherit=False)
    clean.pop('LD_LIBRARY_PATH', None)
    closure = dependencies(lib / 'libhsa-runtime64.so.1', clean)
    files = {'libhsa-runtime64.so.1': hsa}
    for name, path in closure.items():
        if path.is_relative_to(lib.resolve()):
            files[name] = path
    bundled = []
    for name, source in files.items():
        relative = Path('rocm_sysdeps/lib') if source.parent.name == 'lib' and \
            source.parent.parent.name == 'rocm_sysdeps' else Path('.')
        destination = package / 'lib' / relative / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f'Conflicting bundled HSA runtime file: {destination}')
        shutil.copy2(source, destination)
        destination.chmod(destination.stat().st_mode | 0o200)
        bundled.append({'soname': name, 'file': str(Path('lib') / relative / name),
                        'source': str(source), 'source_sha256': digest(source),
                        'sha256': digest(destination)})
    alias = package / 'lib' / 'libhsa-runtime64.so'
    if not alias.exists():
        alias.symlink_to('libhsa-runtime64.so.1')
    licenses = package / 'licenses' / 'hsa-runtime'
    licenses.mkdir(parents=True, exist_ok=True)
    for component in ('rocr', 'rocprofiler-register'):
        notice = root / 'share/doc' / component / 'LICENSE.md'
        if notice.is_file():
            shutil.copy2(notice, licenses / f'{component}-LICENSE.md')
    names = sorted({item['soname'] for item in bundled if 'rocm_sysdeps' in item['file']})
    lines = [SYSDEPS_LICENSES[key] for key in SYSDEPS_LICENSES
             if any(n.startswith(f'librocm_sysdeps_{key}.so') for n in names)]
    (licenses / 'rocm_sysdeps.txt').write_text(
        'TheRock rocm_sysdeps libraries bundled with the HSA runtime:\n' +
        ''.join(f'  {n}\n' for n in names) + '\nComponent licenses:\n' +
        ''.join(f'  {line}\n' for line in lines))
    return bundled


def verify_hsa_runtime(package_lib):
    """The bundled runtime must resolve every ROCm dependency inside the package."""
    clean = runtime_env([], inherit=False)
    clean.pop('LD_LIBRARY_PATH', None)
    links = dependencies(package_lib / 'libhsa-runtime64.so.1', clean)
    for name, path in links.items():
        if name.startswith(('librocm_sysdeps_', 'librocprofiler-register')) and \
                not path.is_relative_to(package_lib.resolve()):
            raise RuntimeError(f'Bundled HSA runtime loads {name} from outside the package: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--loom-manifest', type=Path, required=True)
    parser.add_argument('--package-dir', type=Path)
    parser.add_argument('--hsa-runtime', type=Path,
                        help='ROCm or TheRock root whose lib/ holds the HSA runtime to bundle '
                             '(ROCr 1.21 or newer, with hsa_amd_queue_create); required with --package-dir')
    args = parser.parse_args()
    build = args.build.resolve(strict=True)
    manifest_path = args.loom_manifest.resolve(strict=True)
    compiler = json.loads(manifest_path.read_text())
    library = Path(compiler['library']).resolve(strict=True)
    cache = cmake_cache(build)
    if Path(cache['LSE_LOOMC_LIBRARY']).resolve() != library:
        raise RuntimeError('LSE was not linked with the built, patched Loom compiler')
    if Path(cache['LSE_LOOMC_INCLUDE_DIR']).resolve() != Path(compiler['include_directory']).resolve():
        raise RuntimeError('LSE was compiled with different Loom C API headers')
    if Path(cache['LSE_LOOMC_VERSION_FILE']).resolve() != Path(compiler['version_file']).resolve():
        raise RuntimeError('LSE was configured with different Loom version metadata')
    if digest(library) != compiler['library_sha256']:
        raise RuntimeError('Built Loom compiler differs from its provenance manifest')
    hrx = Path(cache['LSE_HRX_LIBRARY']).resolve(strict=True)
    runtime = Path(cache['LSE_ROCM_RUNTIME_DIR']) if cache.get('LSE_ROCM_RUNTIME_DIR') else None
    env = runtime_env([library.parent, runtime, hrx.parent])
    binaries = [build / name for name in ('lse', 'lse-server')]
    closure = verify(binaries, library, env)
    print(f'Verified selected compiler: {library} SHA256 {digest(library)}')
    if args.package_dir is None:
        (manifest_path.parent / 'LINKED.json').write_text(json.dumps(closure, indent=2) + '\n')
        return

    if args.hsa_runtime is None:
        raise RuntimeError('--hsa-runtime is required to package: the archive bundles its HSA runtime')
    package_arg = args.package_dir.absolute()
    if package_arg.is_symlink():
        raise RuntimeError('Package directory must not be a symlink')
    package = package_arg.resolve(strict=True)
    for name in ('lib', 'libexec', 'bin', 'licenses'):
        (package / name).mkdir(exist_ok=True)
    sources = {soname(library): library, soname(hrx): hrx}
    for links in closure.values():
        for name, path_text in links.items():
            path = Path(path_text)
            if path.is_relative_to(hrx.parent) and name not in sources:
                sources[name] = path
            # A GCC runtime from outside the system library directories (a
            # toolchain newer than the build host's) is bundled, so the archive
            # runs where the system libstdc++ is older.
            if name.startswith(('libstdc++.so', 'libgcc_s.so')) and name not in sources and \
                    not str(path).startswith(('/lib/', '/lib64/', '/usr/lib/', '/usr/lib64/')):
                sources[name] = path
    bundled = []
    for name, source in sources.items():
        destination = package / 'lib' / source.name
        if destination.exists() and digest(destination) != digest(source):
            raise RuntimeError(f'Conflicting bundled library: {destination}')
        shutil.copy2(source, destination)
        destination.chmod(destination.stat().st_mode | 0o200)
        aliases = {name, source.name.split('.so')[0] + '.so'}
        for alias in aliases - {destination.name}:
            target = package / 'lib' / alias
            if target.exists() or target.is_symlink():
                raise RuntimeError(f'Conflicting bundled library alias: {target}')
            target.symlink_to(destination.name)
        bundled.append({'soname': name, 'file': destination.name,
                        'source': str(source), 'source_sha256': digest(source)})

    for name in ('lse', 'lse-server'):
        shutil.move(str(package / name), package / 'libexec' / name)
        for directory, parent in ((package, '.'), (package / 'bin', '..')):
            launcher = directory / name
            launcher.write_text('#!/bin/sh\nset -eu\n'
                f'package_root="$(CDPATH= cd -- "$(dirname -- "$0")/{parent}" && pwd)"\n'
                'export LD_LIBRARY_PATH="$package_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
                f'exec "$package_root/libexec/{name}" "$@"\n')
            launcher.chmod(0o755)

    # Remove build-machine paths from the native closure. The launchers select lib/.
    native = list((package / 'libexec').iterdir()) + [package / 'lib' / item['file'] for item in bundled]
    with tempfile.TemporaryDirectory(prefix='lse-linux-rpath-') as temporary:
        script = Path(temporary) / 'strip.cmake'
        script.write_text(''.join(f'file(RPATH_REMOVE FILE [===[{path}]===])\n' for path in native))
        run('cmake', '-P', str(script))
    for item in bundled:
        item['sha256'] = digest(package / 'lib' / item['file'])
    # The HSA runtime goes in after the RPATH strip: it keeps RUNPATH $ORIGIN.
    hsa_bundled = bundle_hsa_runtime(args.hsa_runtime.resolve(strict=True), package)
    verify_hsa_runtime(package / 'lib')

    work = manifest_path.parent
    shutil.copy2(work / 'source/LICENSE', package / 'licenses/HRX-Loom.txt')
    for base, prefix in ((work / 'source/third_party', 'hrx'), (work / 'build/_deps', 'loom-deps')):
        if not base.is_dir():
            continue
        for notice in base.rglob('*'):
            if notice.is_file() and notice.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')):
                destination = package / 'licenses' / prefix / notice.relative_to(base)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(notice, destination)
    (package / 'BUILD.json').write_text(json.dumps({
        'lse_revision': run('git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD').strip(),
        'loom_compiler': compiler,
        'bundled_libraries': bundled,
        'bundled_hsa_runtime': {'source_root': str(args.hsa_runtime.resolve()), 'files': hsa_bundled},
        'hrx_runtime_source_revision': compiler['hrx_revision'],
        'external_runtime_requirements': ['Compatible Linux C/C++ system runtime',
                                          'amdgpu kernel driver with KFD and access to /dev/kfd'],
        'gpu_execution_tested_by_packager': False,
    }, indent=2) + '\n')
    (package / 'QUICKSTART.txt').write_text(
        'Run ./lse or ./lse-server; bin/ contains equivalent launchers.\n'
        'The archive includes its selected HRX runtime, patched Loom compiler and the\n'
        'HSA runtime (ROCr) HRX needs, in lib/. No ROCm install is required: the system\n'
        'needs a compatible Linux C/C++ runtime and the amdgpu kernel driver with KFD,\n'
        'and the user needs access to /dev/kfd and /dev/dri (usually the render and\n'
        'video groups). The launchers put the bundled libraries first.\n'
        'BUILD.json records source pins, patch hashes and bundled library hashes.\n')

    # A different directory, including spaces, catches paths accidentally retained by the package.
    with tempfile.TemporaryDirectory(prefix='lse-linux-relocation-') as temporary:
        relocated = Path(temporary) / 'relocated package'
        relocated.mkdir()
        for directory in ('lib', 'libexec', 'bin'):
            shutil.copytree(package / directory, relocated / directory, symlinks=True)
        for name in ('lse', 'lse-server'):
            shutil.copy2(package / name, relocated / name)
        verify_hsa_runtime(relocated / 'lib')
        relocated_env = runtime_env([relocated / 'lib', runtime])
        selected = relocated / 'lib' / library.name
        relocated_closure = verify([relocated / 'libexec' / name for name in ('lse', 'lse-server')],
                                   selected, relocated_env)
        for name in ('lse', 'lse-server'):
            run(str(relocated / name), '--help', env=relocated_env)
            run(str(relocated / 'bin' / name), '--help', env=relocated_env)
        for links in relocated_closure.values():
            for name in sources:
                if links.get(name) != str((relocated / 'lib' / sources[name].name).resolve()):
                    raise RuntimeError(f'Relocated binary loads an external bundled dependency: {name}')
    print('Relocated Linux launchers and bundled compiler/HRX/HSA closure passed --help and ldd checks.')


if __name__ == '__main__':
    main()
