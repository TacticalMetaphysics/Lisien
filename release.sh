#!/bin/bash
set -euxo
python3.12 eqversion.py
python -m build --version
python -m twine --version
pyclean --version
wine --version
ls ~/lise_windows
python -m tox
rm -rf LiSE/build LiSE/dist
python -m build LiSE/
rm -rf ELiDE/build ELiDE/dist
python -m build ELiDE/
python -m twine check LiSE/dist/* ELiDE/dist/*
python -m twine upload LiSE/dist/*
python -m twine upload ELiDE/dist/*
WINEPREFIX=~/.wine32 WINEARCH=win32 wine ~/lise_windows/python/python.exe -m pip install --upgrade LiSE ELiDE
pyclean ~/lise_windows
python3.12 butler.py
