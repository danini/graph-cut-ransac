"""Setup for pygcransac."""
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

try:
    from skbuild import setup
except ImportError:
    print("Please update pip to pip 10 or greater, or a manually install the PEP 518 requirements in pyproject.toml", file=sys.stderr)
    raise

cmake_args = []
debug = False
cfg = 'Debug' if debug else 'Release'
cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
cmake_args += ['-DCREATE_SAMPLE_PROJECT=OFF']  # <-- Disable the sample project

setup(
    name='pygcransac',
    version='0.2.dev0',
    author='Daniel Barath, Dmytro Mishkin',
    author_email='barath.daniel@sztaki.mta.hu',
    description='Graph-Cut RANSAC',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages('src'),
    package_dir={'':'src'},
    zip_safe=False,
    include_package_data=False,
    cmake_args=cmake_args,
    cmake_install_dir="src/pygcransac",
    cmake_install_target='install',
    install_requires="numpy",
)
