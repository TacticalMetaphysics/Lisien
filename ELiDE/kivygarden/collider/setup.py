from distutils.core import setup
from distutils.extension import Extension
#import Cython.Compiler.Options
#Cython.Compiler.Options.annotate = True
from Cython.Distutils import build_ext

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("collider", ["collider.pyx"])])
