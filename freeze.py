import sys
if sys.version_info[0] < 3 or (
        sys.version_info[0] == 3 and
        sys.version_info[1] < 3
):
    raise RuntimeError("LiSE requires Python 3.3 or later")
from cx_Freeze import setup, Executable

PY_VERSION_STR = '{}.{}'.format(sys.version_info[0], sys.version_info[1])
PY_STR = 'python' + PY_VERSION_STR

base = None
if sys.platform == "win32":
    base = "Win32GUI"

build_exe_options = {
    "packages": [
        "cython",
        "ELiDE",
        "LiSE",
        "numpy",
        "kivy",
        "pygments",
        "ELiDE.kivygarden.collider",
        "ELiDE.kivygarden.texturestack",
        "configparser",
        "multiprocessing",
        "sqlite3",
        "gorm.xjson"
    ],
    "include_files": [
        ('LiSE/sqlite.json', 'LiSE/sqlite.json'.format(PY_STR)),
        ('ELiDE/assets', 'ELiDE/assets'.format(PY_STR)),
        ('../kivy/kivy/data/style.kv', 'kivy/data/style.kv'.format(PY_STR)),
        ('../kivy/kivy/data/images', 'kivy/data/images'.format(PY_STR)),
        ('../kivy/kivy/data/glsl', 'kivy/data/glsl'.format(PY_STR)),
        ('../kivy/kivy/data/fonts', 'kivy/data/fonts'.format(PY_STR))
    ]
}

setup(
    name="LiSE DevKit",
    version="0.0.0.3",
    description="Standalone package containing the core Life Simulator Engine and the Extensible LiSE Development Environment ELiDE.",
    options={"build_exe": build_exe_options},
    executables=[Executable("start.py", base=base)]
)
