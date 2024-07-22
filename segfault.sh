#!/bin/sh
python -m venv test_segfault
. test_segfault/bin/activate
python -m pip install -r LiSE/test_requirements.txt
PYTHONPATH="$PWD/LiSE:$PWD/ELiDE" python -m pytest ELiDE/
deactivate