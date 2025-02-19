import os
import sys
from multiprocessing import freeze_support

wd = os.getcwd()
sys.path.extend([wd + "/lisien", wd + "/elide"])


def get_application_config(*args):
	return wd + "/elide.ini"


if __name__ == "__main__":
	freeze_support()

	from elide.app import ELiDEApp

	app = ELiDEApp()
	app.get_application_config = get_application_config
	app.run()
