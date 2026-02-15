#!/bin/bash
set -euxo
dos2unix -V
alias python=/usr/local/bin/python3.14
python -m build --version
twine --version
python -m sphinx --version
isort --version
ruff --version
pyclean --version
wine git --version
buildozer --version
JAVA_HOME=$PWD/jdk-17.0.2
export JAVA_HOME
echo "JAVA_HOME=$JAVA_HOME"
javac --version
if [ -n "$(git clean -dn)" ]; then
  echo "Debris in the repository."
  git clean -dn
  exit 1
fi
if [ -n "${CC+x}" ]; then
  echo "Nonstandard C compiler selected."
  echo "$CC"
  exit 1
fi
VERSION=$(python check_version.py)
export VERSION
git submodule init
git submodule update
if [ ! -e pages/lisien-windows.tar.xz ]; then
  echo "lisien-windows.tar.xz not found"
  exit 1
fi
isort lisien
isort elide
ruff format lisien
ruff format elide
pyclean --debris=tox .
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx . pages/docs/
if [ ! -d lisien_windows ]; then
  mkdir lisien_windows
fi
tar -C lisien_windows -xf pages/lisien-windows.tar.xz
rm -rf bin lisien/dist elide/dist
buildozer android clean update debug
wine lisien_windows/python/python.exe -m pip install --force-reinstall lisien/ elide/ 'parquetdb @ git+https://github.com/lllangWV/ParquetDB.git'
pyclean lisien_windows
cp -rf pages/docs lisien_windows/
python -m build lisien/
python -m build elide/
twine check lisien/dist/* elide/dist/*
unix2dos -n CHANGES.txt lisien_windows/CHANGES.txt
cd pages/docs
git add .
git commit -m "Release v${VERSION}"
git tag -f "v${VERSION}"
git push ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
git push --tags ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
cd ../..
git commit -am "Release v${VERSION}"
git tag -f "v${VERSION}"
git push
TWINE_USERNAME=$PYPI_USERNAME TWINE_PASSWORD=$PYPI_PASSWORD twine upload lisien/dist/* elide/dist/*
TWINE_USERNAME=$CODEBERG_USERNAME TWINE_PASSWORD=$CODEBERG_PASSWORD twine upload --repository https://codeberg.org/api/packages/clayote/pypi lisien/dist/* elide/dist/*
python butler.py
