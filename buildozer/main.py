import os

wd = os.getcwd()


def get_application_config(*args):
	return wd + "/elide.ini"


if __name__ == "__main__":
	from elide.app import ElideApp

	connect_string = f"sqlite:///{wd}/world.sqlite3"

	print("connecting to " + connect_string)

	app = ElideApp(path=wd, use_thread=False, connect_string=connect_string)
	app.get_application_config = get_application_config
	app.run()
