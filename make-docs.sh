set -euxo
python -m sphinx --version
pyclean --version
pyclean --debris=all .
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx . pages/docs/
cd pages/docs
git add .
git commit -m "Regen docs"
git push ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
git push --tags ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
cd ../..