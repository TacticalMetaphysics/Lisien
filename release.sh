#!/bin/bash
set -euxo
python -m tox -p4
rm -rf LiSE/dist
python -m build LiSE/
python -m twine --repository LiSE LiSE/dist/*
rm -rf ELiDE/dist
python -m build ELiDE/
python -m twine --repository ELiDE ELiDE/dist/*
