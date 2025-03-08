import os
import shutil
import tempfile

from lisien.engine import Engine
from lisien.examples.college import install

outpath = os.path.join(
	os.path.abspath(os.path.dirname(__file__)), "college24_premade.tar.xz"
)
if os.path.exists(outpath):
	os.remove(outpath)
with tempfile.TemporaryDirectory() as directory:
	with Engine(
		directory,
		workers=0,
		keep_rules_journal=False,
		commit_interval=1,
		connect_string=f"sqlite:///{directory}/world.sqlite3",
	) as eng:
		install(eng)
		for i in range(24):
			print(i)
			eng.next_turn()
		print("Done simulating.")
	print("Compressing...")
	shutil.make_archive(outpath[:-7], "xztar", directory, ".")
print("All done")
