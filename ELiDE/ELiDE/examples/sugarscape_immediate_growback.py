from inspect import getsource
from multiprocessing import freeze_support
from tempfile import mkdtemp

import networkx as nx
from kivy.clock import Clock
from kivy.lang.builder import Builder
from kivy.properties import BooleanProperty, NumericProperty
from networkx import grid_2d_graph

from ELiDE.game import GameApp, GameScreen, GridBoard


# Added pass to make the code run
def make_grid() -> nx.Graph:
    pass


# Added pass to make the code run
def game_start(engine: "LiSE.Engine") -> None:
    pass
