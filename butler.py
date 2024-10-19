import sys
import os
import tomllib

with open("LiSE/pyproject.toml", "rb") as inf:
	cfg = tomllib.load(inf)

version = cfg["project"]["version"]

for wheel in os.listdir("LiSE/dist"):
	if wheel.endswith(".whl"):
		break
else:
	sys.exit("Couldn't find the LiSE wheel")
os.system(
	f"butler push LiSE/dist/{wheel} clayote/lise:lise-whl --userversion {version}"
)
for wheel in os.listdir("ELiDE/dist"):
	if wheel.endswith(".whl"):
		break
else:
	sys.exit("Couldn't find the ELiDE wheel")
os.system(
	f"butler push ELiDE/dist/{wheel} clayote/lise:elide-whl --userversion-file {version}"
)
os.system(
	f"butler push ~/lise_windows clayote/lise:windows --userversion {version}"
)
