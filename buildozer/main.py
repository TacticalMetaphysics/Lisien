import os
import sys

wd = os.getcwd()
sys.path.extend([wd + "/lisien", wd + "/elide"])


def get_application_config(*args):
	return wd + "/elide.ini"


if __name__ == "__main__":
	from elide.app import ElideApp

	app = ElideApp()
	app.get_application_config = get_application_config
	app.run()
