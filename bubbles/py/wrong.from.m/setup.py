
from distutils.core import setup


# from Cython.Build import cythonize

# setup(
#     ext_modules = cythonize("integx.pyx")
# )


from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = "integx",
    cmdclass = {"build_ext": build_ext},
    ext_modules = [ Extension("integx",
                              ["integx.pyx"],
                              libraries=["m"],
                              extra_compile_args = ["-ffast-math"])
    ]
)
