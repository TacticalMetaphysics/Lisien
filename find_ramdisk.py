import os
import subprocess
import tempfile


diskfree = (
	subprocess.Popen(
		["df", "-t", "tmpfs", "--output=target"], stdout=subprocess.PIPE
	)
	.communicate()[0]
	.decode()
)
for line in diskfree.split():
	if not os.path.exists(line):
		continue
	try:
		works = False
		fn = os.path.join(line, "test.txt")
		with open(fn, "wt") as testfile:
			testfile.write("aoeu")
		with open(fn, "rt") as testfile:
			red = testfile.read()
		if red == "aoeu":
			print(line)
			exit()
	except PermissionError:
		continue
print(tempfile.gettempdir())
