from pythonforandroid.recipe import CythonRecipe


class MsgPackRecipe(CythonRecipe):
	version = "1.1.2"
	url = "https://files.pythonhosted.org/packages/4d/f2/bfb55a6236ed8725a96b0aa3acbd0ec17588e6a2c3b62a93eb513ed8783f/msgpack-1.1.2.tar.gz"
	depends = ["setuptools"]
	call_hostpython_via_targetpython = False

	def apply_patches(self, arch, build_dir=None):
		build_dir = build_dir if build_dir else self.get_build_dir(arch.arch)
		super().apply_patches(arch, build_dir)


recipe = MsgPackRecipe()
