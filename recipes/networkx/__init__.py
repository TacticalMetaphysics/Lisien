from pathlib import Path
import shutil

from pythonforandroid.recipe import PyProjectRecipe


class NetworkXRecipe(PyProjectRecipe):
	version = "3.5"
	url = "https://files.pythonhosted.org/packages/6c/4f/ccdb8ad3a38e583f214547fd2f7ff1fc160c43a75af88e6aec213404b96a/networkx-3.5.tar.gz"
	patches = ["patches/no-compress.patch", "patches/no-tests.patch"]

	def apply_patches(self, arch, build_dir=None):
		build_dir = Path(
			build_dir if build_dir is not None else self.get_build_dir(arch)
		)
		for a_path, dir_names, file_names in build_dir.walk():
			if not a_path.exists():
				continue
			for del_dir in ["examples", "tests"]:
				if del_dir in dir_names:
					shutil.rmtree(a_path.joinpath(del_dir))
		super().apply_patches(arch, str(build_dir))

	install_python_package = PyProjectRecipe.install_wheel


recipe = NetworkXRecipe()
