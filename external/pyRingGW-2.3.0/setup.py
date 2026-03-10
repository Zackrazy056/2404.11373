# -*- coding: utf-8 -*-

import os, re
from pathlib import Path

from setuptools import (
    Extension,
    find_packages,
    setup,
)

from Cython.Build import cythonize
import numpy


__ascii_art__ = """\n\n \u001b[\u001b[38;5;39m
                                         @.
                                        &  @
                                        @  ,
                                        (
                                                       *
                                            &            @
                                       #    @        @
                                       @             .    ,
                                       *    .             @
                                                     @
                                                     ,    &
                                      (     #             @           @
                                      *     @                       @   @
                                      *     &       /
                                            .       @      #       @     @          *
*   @  %       *       @       &     @                     %                      @    &          *    @     &    @     @
                                                    *      *              @      @      @     @
                                             &                    @                        %
                                                                 .&        @   @
                                                   .        @                &
                                             @                   @
                                                   @
                                             *               @  @
                                                   .            &
                                                              %&
                                              *
                                              .
                                              @    @
                                              
                                               @  .
                                               /
                                                 @
\u001b[0m"""

def find_version(path, varname="__version__"):
    """Parse the version metadata variable in the given file.
    """
    with open(path, 'r') as fobj:
        version_file = fobj.read()
    version_match = re.search(
        r"^{0} = ['\"]([^'\"]*)['\"]".format(varname),
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# define extension modules
libraries = [] if os.name == "nt" else ["m"]
compile_args = [
    "-O3",
    "-ffast-math",
]
compiler_directives = {
    "embedsignature": True,
    "language_level": 3,
}
define_macros = []

# enable coverage for cython
if int(os.getenv("CYTHON_LINETRACE", "0")):
    compiler_directives["linetrace"] = True
    define_macros.append(("CYTHON_TRACE", "1"))

ext_modules = cythonize(
    [Extension(
         name="pyRing.{}".format(mod),
         sources=["pyRing/{}.pyx".format(mod)],
         include_dirs=[
             numpy.get_include(),
             "pyRing",
         ],
         define_macros=define_macros,
         extra_compile_args=compile_args,
         language="c",
         libraries=libraries,
     ) for mod in (
         "waveform",
         "likelihood",
         "eob_utils",
     )
    ],
    compiler_directives=compiler_directives,
)

# Get the long description from the relevant file
HERE = Path(__file__).parent
with open(HERE / "pypi_description.rst", encoding='utf-8') as f:
    long_description = f.read()

with open(HERE / "requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

setup(
    # metadata
    name="pyRingGW",
    version=find_version(HERE / "pyRing" / "__init__.py"),
    author='Gregorio Carullo, Walter Del Pozzo, John Veitch',
    author_email='gregorio.carullo@ligo.org, walter.delpozzo@ligo.org, john.veitch@ligo.org',
    # contents
    packages=find_packages(),
    ext_modules=ext_modules,
    entry_points={
        "console_scripts": [
            "pyRing = pyRing.pyRing:main",
        ],
    },
    classifiers=[
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent',
                 'Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
    ],
    description='pyRing: Black hole ringdown data-analysis in time-domain',
    license='MIT',
    long_description=long_description,
    url='https://git.ligo.org/lscsoft/pyring',
    project_urls={
      'Bug Tracker': 'https://git.ligo.org/lscsoft/pyring/-/issues',
      'Documentation': 'https://lscsoft.docs.ligo.org/pyring',
      'Source Code': 'https://git.ligo.org/lscsoft/pyring',
    },
    # requirements
    python_requires='>=3',
    setup_requires=[
        "Cython",
        "numpy",
    ],
    install_requires=requirements,
    extras_require={
        "docs": [
            "kotti_docs_theme",
            "myst-parser",
            "sphinx==3.4.3",
            "jinja2==3.0.0",
        ],
    },
)

try:
    import art
    my_art = art.text2art("            Installed     pyRing") # Return ASCII text (default font)
    print("\u001b[\u001b[38;5;39m{}\u001b[0m".format(my_art))
except: print("* Warning: The `art` package could not be imported. Please consider installing it locally for best visual renditions. The cause of this not being taken care of automatically by the `pyRing` package is that the `art` package is not deployed on conda, hence the conda-build fails and `art` cannot be listed as a requirement for the `pyRing` package.")

print(__ascii_art__)
