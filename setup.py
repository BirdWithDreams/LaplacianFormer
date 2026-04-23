from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='Laplace_subtraction_cuda',
    ext_modules=[
        CUDAExtension(
            name='Laplace_subtraction_cuda',
            sources=[
                os.path.join(this_dir, 'src/wrapper.cpp'),
                os.path.join(this_dir, 'src/LaplaceSubtraction_cuda.cu'),
                os.path.join(this_dir, 'src/LaplaceSubtraction_cuda_kernel.cu'),
                os.path.join(this_dir, 'src/NewtonInverse_cuda_kernel.cu'),
                os.path.join(this_dir, 'src/cuda_version.cu'),
            ],
            include_dirs=[
                os.path.join(this_dir, 'include')
            ],
            define_macros=[('WITH_CUDA', None)],  
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-Wno-unused-function'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--use_fast_math',
                    '-lineinfo',
                    '-Xptxas=-v',
                    '-Xcompiler', '-Wno-unused-function'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
