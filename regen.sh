set -euxo
python -m pip install git+https://github.com/clayote/attrs.git@sphinx-attr-getter-ext
python -m sphinx --version
pyclean --version
pyclean --debris=all .
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx . pages/docs/
cd pages/
python -m pip wheel --no-deps ../lisien/ ../elide/
git add .
git commit -m "Regen docs and wheels"
git push ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
git push --tags ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
cd ../
