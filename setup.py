print("This is just for readthedocs, please don't install it for real")

from setuptools import setup
import sys
# convince Kivy to do a Cythonless "build," so we can import it but it does nothing
sys.environ["NDKPLATFORM"] = 1
sys.environ["LIBLINK"] = 1
sys.environ["READTHEDOCS"] = 1
setup(name="LiSE docs", version="0.9", license="AGPL3+", install_requires=["kivy>=1.10.0"])
