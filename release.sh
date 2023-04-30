#!/bin/bash
set -euxo
python -m tox -p auto
rm -rf LiSE/dist
python -m build LiSE/
rm -rf ELiDE/dist
python -m build ELiDE/
python -m twine check LiSE/dist/* ELiDE/dist/*
python -m twine upload --repository LiSE LiSE/dist/*
python -m twine upload --repository ELiDE ELiDE/dist/*
