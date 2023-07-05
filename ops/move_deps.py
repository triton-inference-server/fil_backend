#!/usr/bin/env python3

import os
import re
import shutil
import subprocess
from pathlib import Path

MISSING_REGEX = re.compile(r"\n\t(.+)\ =>\ not\ found")
FOUND_REGEX = re.compile(r"\n\t(.+)\ =>\ (.+)\ (\(0[xX][0-9a-fA-F]+\))")


def ldd(path):
    """Get output of ldd for given file"""
    ldd_out = subprocess.run(["ldd", path], check=True, capture_output=True, text=True)
    return ldd_out.stdout


def get_missing_deps(ldd_output):
    """Return iterator of missing dependencies in ldd output"""
    for match in MISSING_REGEX.finditer(ldd_output):
        yield match.group(1)


def path_contains(parent, child):
    """Check if first path contains the child path"""
    parent = os.path.abspath(parent)
    child = os.path.abspath(child)
    return parent == os.path.commonpath([parent, child])


def get_deps_map(ldd_output, required_dir=None):
    """Return dictionary mapping library names to paths"""
    deps_map = {}
    for match in FOUND_REGEX.finditer(ldd_output):
        if required_dir is None or path_contains(required_dir, match.group(2)):
            deps_map[match.group(1)] = match.group(2)
    return deps_map


def move_dependencies():
    """Move FIL backend dependencies from conda build environment to install
    directory

    The FIL backend library is built within a a conda environment containing
    all required shared libraries for deploying the backend. This function
    analyzes ldd output to determine what libraries FIL links against in its
    build environment as well as what libraries will be missing in the final
    install location. It then moves missing libraries to the final install
    location and repeats the analysis until it has satisfied as many missing
    dependencies as possible.
    """
    fil_lib = os.getenv("FIL_LIB", "libtriton_fil.so")
    lib_dir = os.getenv("LIB_DIR", "/usr/lib")

    conda_lib_dir = os.getenv("CONDA_LIB_DIR")
    if conda_lib_dir is None:
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix is None:
            raise RuntimeError(
                "Must set CONDA_LIB_DIR to conda environment lib directory"
            )
        conda_lib_dir = os.path.join(conda_prefix, "lib")

    Path(lib_dir).mkdir(parents=True, exist_ok=True)

    # Set RUNPATH to conda lib directory to determine locations of
    # conda-provided dependencies
    subprocess.run(["patchelf", "--set-rpath", conda_lib_dir, fil_lib], check=True)

    ldd_out = ldd(fil_lib)
    expected_missing = set(get_missing_deps(ldd_out))
    deps_map = get_deps_map(ldd_out, required_dir=conda_lib_dir)

    # Set RUNPATH to final dependency directory
    subprocess.run(["patchelf", "--set-rpath", lib_dir, fil_lib], check=True)

    prev_missing = {
        None,
    }
    cur_missing = set()
    while prev_missing != cur_missing:
        prev_missing = cur_missing
        cur_missing = set(get_missing_deps(ldd(fil_lib)))
        for missing_dep in cur_missing:
            try:
                lib_path = deps_map[missing_dep]
            except KeyError:
                continue
            shutil.copy(lib_path, lib_dir)

    remaining = cur_missing - expected_missing
    if remaining != {}:
        print("Could not find the following dependencies:")
        for lib in sorted(remaining):
            print(lib)
    else:
        print("All dependencies found")


if __name__ == "__main__":
    move_dependencies()
