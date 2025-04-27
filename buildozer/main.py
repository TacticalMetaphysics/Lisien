import os

wd = os.getcwd()


def get_application_config(*args):
	return wd + "/elide.ini"


if __name__ == "__main__":
	from elide.app import ElideApp

	app = ElideApp(path=wd, use_thread=True)
	app.get_application_config = get_application_config
	app.run()
