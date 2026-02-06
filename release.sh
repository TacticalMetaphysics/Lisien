#!/bin/bash
set -euxo
dos2unix -V
alias python=python3.12
VERSION=$(python3.12 check_version.py)
export VERSION
python -m build --version
twine --version
python -m sphinx --version
isort --version
tox --version
ruff --version
pyclean --version
wine git --version
buildozer --version
ls /usr/lib/jvm/java-25-openjdk-amd64
if [ -n "$(git clean -n)" ]; then
  echo "Debris in the repository."
  git clean -n
  exit 1
fi
if [ -n "${CC+x}" ]; then
  echo "Nonstandard C compiler selected."
  echo "$CC"
  exit 1
fi
wget https://clayote.codeberg.page/lisien-windows.zip
unzip lisien-windows.zip
isort lisien
isort elide
ruff format lisien
ruff format elide
pyclean --debris=tox .
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx . pages/docs/
rm -rf bin lisien/dist elide/dist
JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 buildozer android update clean debug
cd pages/docs
git add .
git commit -m "Release v${VERSION}"
git push
git push --tags
cd ../..
git commit -am "Release v${VERSION}"
git tag -f "v${VERSION}"
git push
git push --tags
python -m build lisien/
python -m build elide/
twine check lisien/dist/* elide/dist/*
twine upload lisien/dist/* elide/dist/*
twine upload --repository codeberg lisien/dist/* elide/dist/*
wine lisien_windows/python/python.exe -m pip install --force-reinstall lisien/ elide/ 'parquetdb @ git+https://github.com/lllangWV/ParquetDB.git'
pyclean lisien_windows
unix2dos -n CHANGES.txt lisien_windows/CHANGES.txt
cp -rf pages/docs lisien_windows/
python3.12 butler.py
