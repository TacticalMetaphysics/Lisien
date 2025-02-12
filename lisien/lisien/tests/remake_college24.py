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
	# Apparently, keeping the rules journal causes it to get all backed up
	# and thrash the heck out of the database on engine close.
	# I don't *need* that journal for this case, but I'd better make it
	# perform better for when I do.
	with Engine(directory, workers=0, keep_rules_journal=False) as eng:
		install(eng)
		for i in range(24):
			print(i)
			eng.next_turn()
	print("done simulating. Compressing...")
	shutil.make_archive(outpath[:-7], "xztar", directory, ".")
# But also, sometimes it hangs after the archive's been created. Can't tell why...
print("done")
