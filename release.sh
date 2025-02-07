#!/bin/bash
set -euxo
dos2unix -V
python3.12 eqversion.py
python -m build --version
python -m twine --version
python -m sphinx --version
pyclean --version
wine --version
ls ~/lise_windows
rm -rf .tox
python -m tox
rm -rf .tox
python -m sphinx . docs/
rm -rf lisien/build lisien/dist
python -m build lisien/
rm -rf elide/build elide/dist
python -m build elide/
python -m twine check lisien/dist/* elide/dist/*
python -m twine upload lisien/dist/* elide/dist/*
python -m twine upload --repository codeberg lisien/dist/* elide/dist/*
WINEPREFIX=~/.wine32 WINEARCH=win32 wine ~/lisien_windows/python/python.exe -m pip install --upgrade lisien elide
pyclean ~/lise_windows
unix2dos -n CHANGES.txt ~/lisien_windows/CHANGES.txt
cp -rf docs ~/lisien_windows/
python3.12 butler.py
