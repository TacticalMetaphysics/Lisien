import os
import tempfile
import shutil

from lisien.engine import Engine
from lisien.examples.college import install

outpath = os.path.join(
	os.path.abspath(os.path.dirname(__file__)), "college24_premade.tar.xz"
)
if os.path.exists(outpath):
	os.remove(outpath)
with tempfile.TemporaryDirectory() as directory:
	with Engine(directory) as eng:
		install(eng)
		for i in range(24):
			eng.next_turn()
	shutil.make_archive(outpath[:-7], "xztar", directory, ".")
