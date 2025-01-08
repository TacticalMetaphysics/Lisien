"""
This script automates the process of pushing the LiSE and ELiDE wheel files to the Butler repository.
It reads the project version from the pyproject.toml file and uses it to tag the pushed files.
"""

import os
import sys
import tomllib

with open("LiSE/pyproject.toml", "rb") as inf:
    cfg = tomllib.load(inf)

version = cfg["project"]["version"]
# Initialized wheel before loop
WHEEL = None

for WHEEL in os.listdir("LiSE/dist"):
    if WHEEL.endswith(".whl"):
        break
else:
    sys.exit("Couldn't find the LiSE wheel")
os.system(
    f"butler push LiSE/dist/{WHEEL} clayote/lise:lise-whl --userversion {version}"
)
for WHEEL in os.listdir("ELiDE/dist"):
    if WHEEL.endswith(".whl"):
        break
else:
    sys.exit("Couldn't find the ELiDE wheel")
os.system(
    f"butler push ELiDE/dist/{WHEEL} clayote/lise:elide-whl --userversion {version}"
)
os.system(f"butler push ~/lise_windows clayote/lise:windows --userversion {version}")