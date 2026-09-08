"""Build and register the native CUDA library without changing the Python environment."""
import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import jax

@functools.lru_cache(None)
def register():
    source = Path(__file__).with_name('cutlass_block.cu')
    cuda = Path(os.environ.get('BRC_CUDA_ROOT', '/usr/local/cuda-12.8'))
    cutlass = Path(os.environ.get('BRC_CUTLASS_ROOT', '/home/caiyuliang/brc_v1_audit/cutlass'))
    cutlass_sha = subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    version = subprocess.check_output([str(cuda/'bin/nvcc'), '--version'], text=True)
    digest = hashlib.sha256(source.read_bytes() + Path(__file__).read_bytes() + version.encode() + jax.__version__.encode() + cutlass_sha.encode()).hexdigest()
    cache = Path(os.environ.get('BRC_KERNEL_CACHE', '/home/caiyuliang/brc_v1_audit/build'))/'sm120'/digest
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache/'brc_block.so'
    if not lib.exists():
        command = [str(cuda/'bin/nvcc'), '-std=c++17', '--expt-relaxed-constexpr', '-w', '-O3', '-shared', '-Xcompiler', '-fPIC', '-arch=sm_120a', str(source), '-I'+jax.ffi.include_dir(), '-L'+str(cuda/'lib64'), '-I'+str(cutlass/'include'), '-I'+str(cutlass/'tools/util/include'), '-lcudart', '-Xlinker', '-rpath,'+str(cuda/'lib64'), '-o', str(lib)]
        subprocess.run(command, check=True)
        (cache/'manifest.json').write_text(json.dumps({'hash':digest,'command':command,'cuda':version,'jax':jax.__version__,'cutlass':cutlass_sha}, indent=2))
    loaded = ctypes.CDLL(str(lib))
    jax.ffi.register_ffi_target('brc_mxfp8', jax.ffi.pycapsule(loaded.brc_mxfp8), platform='CUDA')
    return loaded, digest
