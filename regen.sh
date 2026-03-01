set -euxo
pyclean --version
python -m sphinx --version
python -m pip install hatch
git clone --depth 1 -b sphinx-attr-getter-ext https://github.com/clayote/attrs.git
patch -d attrs -p1 <attrs.patch
SETUPTOOLS_SCM_PRETEND_VERSION=0.22.4 uv build --wheel attrs/
python -m pip install attrs-*.whl
pyclean --debris=all .
PYTHONPATH=$PWD/lisien:$PWD/elide python -m sphinx . pages/docs/
cd pages/
python -m pip wheel --no-deps ../lisien/ ../elide/
git add .
git commit -m "Regen docs and wheels"
git push ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
git push --tags ssh://git@codeberg.org/clayote/Lisien.git HEAD:pages
cd ../
