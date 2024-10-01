from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sysconfig

__version__ = "1.0.0"
# Get the path to the active conda environment
conda_prefix = os.getenv('CONDA_PREFIX')
print(conda_prefix)
# Common flags for both release and debug builds.
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
#extra_compile_args += ["-std=c++23"]

ext_modules = [
    Pybind11Extension(
        "Topograph",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        #["src/_topograph.cpp"],
        include_dirs=[
            os.path.join(conda_prefix, 'include', 'opencv4'),
            os.path.join("..", 'eigen4'),
            os.path.join("..", 'boost_1_82_0'),
        ],
        library_dirs=[os.path.join(conda_prefix, 'lib')],
        libraries=['opencv_core', 'opencv_imgproc'],
        #library_dirs=[os.path.join(conda_prefix, 'lib')],
        #libraries=['opencv_core', 'opencv_imgproc'],
        extra_compile_args=["-fopenmp"],
    ),
]

setup(name="Topograph",
    version=__version__,
    author="Alexander H. Berger",
    author_email="a.berger@tum.de",
    url="",
    description="Highly Efficient C++ Topograph implementation",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7"
)