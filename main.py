import os
import sys

try:
	from multiprocessing import freeze_support
except ImportError:

	def freeze_support(): ...


wd = os.getcwd()
sys.path.extend([wd, wd + "/lisien", wd + "/elide"])


def get_application_config(*args):
	return wd + "/elide.ini"


if __name__ == "__main__":
	freeze_support()

	from elide.app import ElideApp

	app = ElideApp(prefix=wd, connect_string=f"sqlite:///{wd}/world.sqlite3")
	app.get_application_config = get_application_config
	app.run()
