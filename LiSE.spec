# -*- mode: python -*-
from kivy.tools.packaging.pyinstaller_hooks import (
    get_deps_minimal, 
    hookspath,
    runtime_hooks
)
import os
a = Analysis(['start.py'],
             pathex=['/home/sanotehu/src/LiSE'],
             hiddenimports=[
                 'kivy.graphics.instructions',
                 'kivy.graphics.buffer',
                 'kivy.graphics.vertex',
                 'kivy.graphics.vbo',
                 'kivy.graphics.compiler',
                 'kivy.graphics.shader',
                 'kivy.core.image._img_sdl2',
                 'kivy.core.image.img_sdl2',
                 'kivy.core.text.text_sdl2',
                 'kivy.core.window.window_sdl2',
                 'kivy.core.window._window_sdl2',
                 'pygments.formatters.bbcode',
                 'ELiDE.kivygarden.collider',
                 'ELiDE.kivygarden.texturestack',
                 'ELiDE.kivygarden.stiffscroll'
             ],
             hookspath=hookspath(),
             runtime_hooks=runtime_hooks(),
             **get_deps_minimal(video=None, audio=None)
             )
collider_built = os.listdir('ELiDE/kivygarden/collider/build')
for fn in collider_built:
    if fn.startswith('lib'):
        collider_dir = 'ELiDE/kivygarden/collider/build/' + fn
        try:
            collider_lib = os.listdir(collider_dir)[0]
        except IndexError:
            exit("Couldn't find collider lib")
        collider_lib_path = collider_dir + '/' + collider_lib
        break
else:
    exit("Couldn't find collider lib")
a.binaries += [
    ('ELiDE/kivygarden/collider/' + collider_lib,
     collider_lib_path,
     'BINARY')
]
a.datas += [
    ('ELiDE/elide.kv', 'ELiDE/elide.kv', 'DATA'),
    ('kivy/data/style.kv', '../kivy/kivy/data/style.kv', 'DATA'),
    ('LiSE/sqlite.json', 'LiSE/sqlite.json', 'DATA')
] + Tree('../kivy/kivy/data/images', prefix='kivy/data/images') \
    + Tree('../kivy/kivy/data/glsl', prefix='kivy/data/glsl') \
    + Tree('../kivy/kivy/data/fonts', prefix='kivy/data/fonts') \
    + Tree('ELiDE/assets', prefix='ELiDE/assets')
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='start',
          debug=False,
          strip=None,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='start')
