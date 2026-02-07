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
JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64
ls $JAVA_HOME
export JAVA_HOME
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
VERSION=$(python check_version.py)
export VERSION
wget https://clayote.codeberg.page/lisien-windows.zip
isort lisien
isort elide
ruff format lisien
ruff format elide
pyclean --debris=tox .
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx . pages/docs/
unzip lisien-windows.zip
rm -rf bin lisien/dist elide/dist
buildozer android update clean debug
wine lisien_windows/python/python.exe -m pip install --force-reinstall lisien/ elide/ 'parquetdb @ git+https://github.com/lllangWV/ParquetDB.git'
pyclean lisien_windows
cp -rf pages/docs lisien_windows/
python -m build lisien/
python -m build elide/
twine check lisien/dist/* elide/dist/*
mv CHANGES.txt CHANGES.txt.old
echo "$LISIEN_CHANGES" | cat - CHANGES.txt.old >CHANGES.txt
unix2dos -n CHANGES.txt lisien_windows/CHANGES.txt
cd pages/docs
git add .
git commit -m "Release v${VERSION}"
git tag -f "v${VERSION}"
git push
git push --tags
cd ../..
git commit -am "Release v${VERSION}"
git tag -f "v${VERSION}"
git push
git push --tags
TWINE_USERNAME=$CODEBERG_USERNAME TWINE_PASSWORD=$CODEBERG_PASSWORD twine upload lisien/dist/* elide/dist/*
TWINE_USERNAME=$PYPI_USERNAME TWINE_PASSWORD=$PYPI_PASSWORD twine upload --repository https://codeberg.org/api/packages/clayote/pypi lisien/dist/* elide/dist/*
python butler.py
