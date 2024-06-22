#!/bin/bash
set -euxo
python -m tox -p auto
rm -rf LiSE/build LiSE/dist
python -m build LiSE/
rm -rf ELiDE/build ELiDE/dist
python -m build ELiDE/
python -m twine check LiSE/dist/* ELiDE/dist/*
python -m twine upload --repository LiSE LiSE/dist/*
python -m twine upload --repository ELiDE ELiDE/dist/*
rm -rf LiSE/build LiSE/dist ELiDE/build ELiDE/dist
