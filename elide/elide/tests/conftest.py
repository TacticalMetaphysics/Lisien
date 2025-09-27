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
from kivy.lang import Builder
from kivy.resources import resource_find

from elide.app import ElideApp
from elide.tests.util import idle_until
from elide.util import unload_all_kv
from lisien import Engine
from lisien.examples import kobold, polygons, sickle


@pytest.fixture
def prefix(tmp_path):
	os.makedirs(
		os.path.join(
			tmp_path,
			"games",
		)
	)
	os.makedirs(
		os.path.join(
			tmp_path,
			"test",
		)
	)
	yield str(tmp_path)


@pytest.fixture()
def random_seed():
	yield 69105


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


def make_elide_app(play_dir, **kwargs):
	Builder.load_file(resource_find("elide.kv"))
	return ElideApp(
		prefix=play_dir, games_dir="games", game_name="test", **kwargs
	)


@pytest.fixture
def elide_app(kivy, prefix):
	app = make_elide_app(
		prefix, immediate_start=True, character_name="physical", workers=0
	)
	app.leave_game = True
	app.config = ConfigParser(None)
	app.build_config(app.config)
	Window.add_widget(app.build())
	yield app
	EventLoop.idle()
	if not app.stopped:
		app.stop()
	unload_all_kv()


@pytest.fixture(scope="function")
def elide_app_main_menu(kivy, prefix):
	app = make_elide_app(prefix, workers=0)
	app.config = ConfigParser(None)
	app.build_config(app.config)
	Window.add_widget(app.build())
	manager = app.manager
	idle_until(
		lambda: manager.current == "main",
		100,
		"Never switched to 'main' screen",
	)
	idle_until(
		lambda: manager.current_screen.ids, 100, "Never got manager.ids"
	)
	for button in manager.current_screen.ids.values():
		idle_until(
			lambda: button.pos != [0, 0],
			100,
			"Never laid out the buttons",
		)
	yield app
	EventLoop.idle()
	if not app.stopped:
		app.stop()


@pytest.fixture
def line_shaped_graphs(prefix):
	with Engine(os.path.join(prefix, "test")) as eng:
		eng.add_character("physical", nx.grid_2d_graph(10, 1))
		eng.add_character("tall", nx.grid_2d_graph(1, 10))


@pytest.fixture
def sickle_sim(prefix, random_seed):
	with Engine(
		os.path.join(prefix, "test"), workers=0, random_seed=random_seed
	) as eng:
		sickle.install(eng)


@pytest.fixture
def kobold_sim(prefix, random_seed):
	play_prefix = os.path.join(prefix, "test")
	with Engine(play_prefix, workers=0, random_seed=random_seed) as eng:
		kobold.inittest(eng)
	yield play_prefix


@pytest.fixture
def kobold_sim_exported(tmp_path, kobold_sim):
	with Engine(kobold_sim, workers=0) as eng:
		exported = eng.export(
			"kobold", os.path.join(tmp_path, "kobold.lisien")
		)
	yield exported


@pytest.fixture
def polygons_sim(prefix, random_seed):
	with Engine(
		os.path.join(prefix, "test"), workers=0, random_seed=random_seed
	) as eng:
		polygons.install(eng)
