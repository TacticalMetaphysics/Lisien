import sh

from multiprocessing import cpu_count

from pythonforandroid.archs import Arch
from pythonforandroid.logger import shprint
from pythonforandroid.recipe import Recipe
from pythonforandroid.util import current_directory


class LibZstdRecipe(Recipe):
	version = "1.5.7"
	url = "https://github.com/facebook/zstd/releases/download/v{version}/zstd-{version}.tar.gz"
	built_libraries = {"libzstd.so": "lib"}

	def build_arch(self, arch: Arch):
		env = self.get_recipe_env(arch)
		with current_directory(self.get_build_dir(arch.arch)):
			shprint(
				sh.make,
				"-j",
				str(cpu_count()),
				f"CC={env['CC']}",
				"-f",
				"Makefile",
				_env=env,
			)

	def get_library_includes(self, arch: Arch) -> str:
		return " -I" + self.get_build_dir(arch.arch) + "/lib"

	def get_library_ldflags(self, arch: Arch) -> str:
		return " -L" + self.get_build_dir(arch.arch) + "/lib"

	@staticmethod
	def get_library_libs_flag() -> str:
		return " -lzstd"


recipe = LibZstdRecipe()
