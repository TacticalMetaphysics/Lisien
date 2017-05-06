# -*- mode: python -*-
from kivy.tools.packaging.pyinstaller_hooks import hookspath, runtime_hooks, get_deps_minimal
block_cipher = None

mindeps = get_deps_minimal(audio=None, video=None)
mindeps['hiddenimports'].extend([
	'ELiDE.kivygarden.collider',
	'ELiDE.kivygarden.stiffscroll',
	'ELiDE.kivygarden.texturestack',
	'kivy.weakmethod'
])

a = Analysis(['main.py'],
             pathex=['LiSE', 'ELiDE', 'allegedb'],
             binaries=[('ELiDE/ELiDE/kivygarden/collider/build/lib.macosx-10.6-intel-3.5/collider.cpython-35m-darwin.so', '.')],  # produced by python3 setup.py build_ext in the collider/ directory
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
          console=True )
coll = COLLECT(exe,
               Tree('/Library/Frameworks/SDL2_ttf.framework/Versions/A/Frameworks/FreeType.framework'),
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='ELiDEa7')
