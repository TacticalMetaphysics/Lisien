# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import sys

try:
	from multiprocessing import freeze_support
except ImportError:

	def freeze_support(): ...


try:
	from android.storage import app_storage_path

	wd = app_storage_path()
except ImportError:
	wd = os.getcwd()
sys.path.extend([wd, wd + "/lisien", wd + "/elide"])


def get_application_config(*args):
	return wd + "/elide.ini"


if __name__ == "__main__":
	freeze_support()

	from kivy.logger import Logger
	from elide.app import ElideApp

	Logger.setLevel(10)

	app = ElideApp(
		prefix=wd,
		connect_string=f"sqlite:///{wd}/world.sqlite3",
	)
	app.get_application_config = get_application_config
	app.run()
