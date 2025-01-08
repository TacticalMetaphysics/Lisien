"""
This script checks if the version numbers of the LiSE and ELiDE projects are the same.
It reads the version numbers from the pyproject.toml files of both projects and exits with an error message if they differ.
"""

import sys
import tomllib

with open("LiSE/pyproject.toml", "rb") as inf:
    lise_version = tomllib.load(inf)["project"]["version"]
with open("ELiDE/pyproject.toml", "rb") as inf:
    elide_version = tomllib.load(inf)["project"]["version"]

if lise_version != elide_version:
    sys.exit(f"Version numbers differ. LiSE: {lise_version}, ELiDE: {elide_version}")
