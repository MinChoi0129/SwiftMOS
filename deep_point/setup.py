from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-Xcompiler=-fPIC",
    # ===== Blackwell+ (CC 12.0) 전용 =====
    "-gencode=arch=compute_120,code=sm_120",
    # 선택: PTX도 함께 넣어서 향후 드라이버 JIT 여지 주기
    "-gencode=arch=compute_120,code=compute_120",
    "-Wno-deprecated-gpu-targets",
]

setup(
    name="deep_point",
    version="1.1",
    description="deep layers used to convert between point and voxels",
    author="gang.zhang",
    author_email="zhanggang11021136@gmail.com",
    ext_modules=[
        CppExtension(
            name="point_deep.cpu_kernel",
            sources=["src/point_deep.cpp"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        ),
        CUDAExtension(
            name="point_deep.cuda_kernel",
            sources=["src/point_deep_cuda.cpp", "src/point_deep_cuda_kernel.cu"],
            include_dirs=["src"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": extra_cuda_cflags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
)
