import sys, os
prefix = os.getcwd()
os.chdir('igraph-0.6.5')
os.system('./configure --prefix=' + prefix)
os.system('make')
os.system('make install')
os.chdir(prefix + os.sep + 'python-igraph-0.6.5')
os.system('python setup.py --no-pkg-config build')
os.chdir(prefix)
