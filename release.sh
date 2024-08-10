#!/bin/bash
set -euxo
python -m tox
rm -rf LiSE/build LiSE/dist
python -m build LiSE/
rm -rf ELiDE/build ELiDE/dist
python -m build ELiDE/
python -m twine check LiSE/dist/* ELiDE/dist/*
python -m twine upload LiSE/dist/*
python -m twine upload ELiDE/dist/*
rm -rf LiSE/build LiSE/dist ELiDE/build ELiDE/dist
