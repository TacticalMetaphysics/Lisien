# -*- mode: python -*-
from kivy.deps import sdl2, glew
from kivy.tools.packaging.pyinstaller_hooks import hookspath, runtime_hooks, get_deps_minimal
block_cipher = None

mindeps = get_deps_minimal(audio=None, video=None)
mindeps['hiddenimports'].extend([
	'ELiDE.kivygarden.texturestack',
	'kivy.weakmethod',
	'umsgpack'
])

a = Analysis(['main.py'],
             pathex=['LiSE', 'ELiDE', 'allegedb'],
             datas=[
                 ('ELiDE/ELiDE/assets', 'ELiDE/assets'),
                 ('LiSE/LiSE/sqlite.json', 'LiSE')
             ],
             hookspath=hookspath(),
             runtime_hooks=runtime_hooks(),
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
			 **mindeps
			 )
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='RunThis',
          debug=False,
          strip=False,
          upx=True,
          console=True,
          icon='ELiDE_icon/Windows/ELiDE.ico'
)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
			   *(Tree(p) for p in sdl2.dep_bins + glew.dep_bins),
               strip=False,
               upx=True,
               name='ELiDE-windows-0.9.1')
