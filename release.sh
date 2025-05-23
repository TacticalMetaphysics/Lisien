#!/bin/bash
set -euxo
dos2unix -V
alias python=python3.12
python3.12 eqversion.py
python -m build --version
twine --version
python -m sphinx --version
isort --version
tox --version
ruff --version
pyclean --version
wine --version
ls ~/lisien_windows
isort .
ruff format .
rm -rf lisien/build lisien/dist lisien/strings
rm -rf elide/build elide/dist elide/strings
tox -pauto -c lisien/tox.ini
rm -rf lisien/build lisien/dist lisien/strings
rm -rf elide/build elide/dist elide/strings
tox -pauto -c elide/tox.ini
rm -rf lisien/build lisien/dist lisien/strings
rm -rf elide/build elide/dist elide/strings
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx . docs/
python -m build lisien/
python -m build elide/
twine check lisien/dist/* elide/dist/*
twine upload lisien/dist/* elide/dist/*
twine upload --repository codeberg lisien/dist/* elide/dist/*
wine ~/lisien_windows/python/python.exe -m pip install --upgrade lisien/ elide/
pyclean ~/lisien_windows
unix2dos -n CHANGES.txt ~/lisien_windows/CHANGES.txt
cp -rf docs ~/lisien_windows/
python3.12 butler.py
