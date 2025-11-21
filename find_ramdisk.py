import os
import subprocess
import tempfile


diskfree = (
	subprocess.Popen(
		["df", "-t", "tmpfs", "--output=avail,target"], stdout=subprocess.PIPE
	)
	.communicate()[0]
	.decode()
)
tmpdirs = []
for line in diskfree.split("\n"):
	try:
		avail, target = line.split()
	except ValueError:
		continue
	tmpdirs.append((int(avail), str(target)))
tmpdirs.sort(reverse=True)
for _, target in tmpdirs:
	if not os.path.exists(target):
		continue
	try:
		works = False
		fn = os.path.join(target, "test.txt")
		with open(fn, "wt") as testfile:
			testfile.write("aoeu")
		with open(fn, "rt") as testfile:
			red = testfile.read()
		if red == "aoeu":
			print(target)
			exit()
	except (OSError, PermissionError):
		continue
print(tempfile.gettempdir())
