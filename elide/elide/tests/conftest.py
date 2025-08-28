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

import networkx as nx
import pytest
from kivy.base import EventLoop, stopTouchApp
from kivy.config import ConfigParser
from kivy.core.window import Window

from elide.app import ElideApp
from lisien import Engine


@pytest.fixture
def play_dir(tmp_path):
	games_dir = "games"
	game_name = "test"
	play_dir = os.path.join(tmp_path, games_dir, game_name)
	os.makedirs(
		play_dir,
		exist_ok=True,
	)
	yield play_dir


@pytest.fixture(params=[69105])
def random_seed(request):
	yield request.param


@pytest.fixture
def kivy():
	def clear_window_and_event_loop():
		for child in Window.children[:]:
			Window.remove_widget(child)
		Window.canvas.before.clear()
		Window.canvas.clear()
		Window.canvas.after.clear()
		EventLoop.touches.clear()
		for post_proc in EventLoop.postproc_modules:
			if hasattr(post_proc, "touches"):
				post_proc.touches.clear()
			elif hasattr(post_proc, "last_touches"):
				post_proc.last_touches.clear()

	from os import environ

	environ["KIVY_USE_DEFAULTCONFIG"] = "1"

	# force window size + remove all inputs
	from kivy.config import Config

	Config.set("graphics", "width", "320")
	Config.set("graphics", "height", "240")
	for items in Config.items("input"):
		Config.remove_option("input", items[0])

	# ensure our window is correctly created
	Window.create_window()
	Window.register()
	Window.initialized = True
	Window.close = lambda *s: None
	clear_window_and_event_loop()

	yield
	if EventLoop.status == "started":
		clear_window_and_event_loop()
		stopTouchApp()


@pytest.fixture
def elide_app(kivy, play_dir):
	game_name = os.path.basename(play_dir)
	games_dir = os.path.basename(play_dir[: -len(game_name) - 1])
	prefix = play_dir[: -(len(games_dir) + len(game_name) + 1)]
	character_name = "physical"
	app = ElideApp(
		immediate_start=True,
		prefix=prefix,
		games_dir=games_dir,
		game_name=game_name,
		character_name=character_name,
		workers=0,
	)
	app.leave_game = True
	app.config = ConfigParser(None)
	app.build_config(app.config)
	Window.add_widget(app.build())
	yield app
	EventLoop.idle()
	if not hasattr(app, "stopped"):
		app.stop()


@pytest.fixture
def line_shaped_graphs(play_dir):
	with Engine(play_dir) as eng:
		eng.add_character("physical", nx.grid_2d_graph(10, 1))
		eng.add_character("tall", nx.grid_2d_graph(1, 10))
