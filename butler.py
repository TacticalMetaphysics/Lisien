import sys
import os
import tomllib

with open("lisien/pyproject.toml", "rb") as inf:
	cfg = tomllib.load(inf)

version = cfg["project"]["version"]

for lisien_wheel in os.listdir("lisien/dist"):
	if lisien_wheel.endswith(".whl"):
		break
else:
	sys.exit("Couldn't find the lisien wheel")
for elide_wheel in os.listdir("elide/dist"):
	if elide_wheel.endswith(".whl"):
		break
else:
	sys.exit("Couldn't find the elide wheel")
os.system(
	f"butler push lisien/dist/{lisien_wheel} clayote/lisien:lisien-whl --userversion {version}"
)
os.system(
	f"butler push elide/dist/{elide_wheel} clayote/lisien:elide-whl --userversion {version}"
)
os.system(
	f"butler push ~/lisien_windows clayote/lisien:windows --userversion {version}"
)
