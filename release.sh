#!/bin/bash
set -euxo
dos2unix -V
alias python=python3.12
python3.12 eqversion.py
python -m build --version
python -m twine --version
python -m sphinx --version
pyclean --version
wine --version
ls ~/lisien_windows
rm -rf .tox
python -m tox
rm -rf .tox
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx . docs/
rm -rf lisien/build lisien/dist
python -m build lisien/
rm -rf elide/build elide/dist
python -m build elide/
python -m twine check lisien/dist/* elide/dist/*
python -m twine upload lisien/dist/* elide/dist/*
python -m twine upload --repository codeberg lisien/dist/* elide/dist/*
wine ~/lisien_windows/python/python.exe -m pip install --upgrade lisien elide
pyclean ~/lisien_windows
unix2dos -n CHANGES.txt ~/lisien_windows/CHANGES.txt
cp -rf docs ~/lisien_windows/
python3.12 butler.py
