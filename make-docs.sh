set -euxo
python -m sphinx --version
pyclean --version
pyclean --debris=all .
if [ -z "${VERSION+x}" ]; then
  VERSION=$(python check_version.py)
fi
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx html . pages/docs/
cd pages/docs
git add .
git commit -m "Regen docs for v${VERSION}"
git tag -f "v${VERSION}"
git push ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
git push --tags ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
cd ../..