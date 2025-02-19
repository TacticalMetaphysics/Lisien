from inspect import getsource
from multiprocessing import freeze_support
from tempfile import mkdtemp

import networkx as nx
from kivy.clock import Clock
from kivy.lang.builder import Builder
from kivy.properties import BooleanProperty, NumericProperty
from networkx import grid_2d_graph

from elide.game import GameApp, GameScreen, GridBoard


def make_grid() -> nx.Graph:
	pass


def game_start(engine: "lisien.Engine") -> None:
	pass
