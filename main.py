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
	from android.storage import app_storage_path

	def freeze_support(): ...

	wd = app_storage_path()
	connect_string = "sqlite:///{prefix}/world.sqlite3"
	logs_dir = os.path.join(wd, "app", ".kivy", "logs")
	android = True
except ImportError:
	from multiprocessing import freeze_support

	wd = os.path.join(os.getcwd())
	logs_dir = connect_string = None
	android = False
sys.path.extend([wd, wd + "/lisien", wd + "/elide"])


if __name__ == "__main__":
	freeze_support()
	from logging import getLogger
	from elide.app import ElideApp
	from lisien.enum import Sub

	Logger = getLogger("kivy")
	Logger.setLevel(10)

	app = ElideApp(
		prefix=wd,
		games_dir="games",
		connect_string=connect_string,
		logs_dir=logs_dir,
		sub_mode=Sub.thread,
		android=android,
	)
	try:
		app.run()
	except BaseException as ex:
		import traceback
		from io import StringIO

		bogus = StringIO()
		traceback.print_exception(ex, file=bogus)
		for line in bogus.getvalue().split("\n"):
			Logger.error(line)
	finally:
		app._copy_log_files()
