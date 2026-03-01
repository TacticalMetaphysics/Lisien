set -euxo
python -m pip install https://github.com/clayote/attrs/archive/refs/heads/sphinx-attr-getter-ext.zip
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
