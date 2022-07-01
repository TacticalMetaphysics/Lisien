import sys
from tempfile import mkdtemp
import shutil
from functools import partial

from blinker import Signal
from kivy.input.motionevent import MotionEvent
from kivy.base import EventLoop
from kivy.tests.common import GraphicUnitTest
from kivy.config import ConfigParser

from LiSE.proxy import RedundantProcessError
from ELiDE.app import ELiDEApp


def all_spots_placed(board, char=None):
	if char is None:
		char = board.character
	for place in char.place:
		if place not in board.spot:
			return False
	return True


def all_pawns_placed(board, char=None):
	if char is None:
		char = board.character
	for thing in char.thing:
		if thing not in board.pawn:
			return False
	return True


def all_arrows_placed(board, char=None):
	if char is None:
		char = board.character
	for orig, dests in char.portal.items():
		if orig not in board.arrow:
			return False
		arrows = board.arrow[orig]
		for dest in dests:
			if dest not in arrows:
				return False
	return True


def board_is_arranged(board, char=None):
	if char is None:
		char = board.character
	return all_spots_placed(board, char) and all_pawns_placed(
		board, char) and all_arrows_placed(board, char)


def idle_until(condition=None, timeout=100, message="Timed out"):
	"""Advance frames until ``condition()`` is true

    With integer ``timeout``, give up after that many frames,
    raising ``TimeoutError``. You can customize its ``message``.

    """
	if not (timeout or condition):
		raise ValueError("Need timeout or condition")
	if condition is None:
		return partial(idle_until, timeout=timeout, message=message)
	if timeout is None:
		while not condition():
			EventLoop.idle()
		return
	for _ in range(timeout):
		if condition():
			return
		EventLoop.idle()
	raise TimeoutError(message)


def window_with_widget(wid):
	EventLoop.ensure_window()
	win = EventLoop.window
	win.add_widget(wid)
	return win


class MockTouch(MotionEvent):

	def depack(self, args):
		self.is_touch = True
		self.sx = args['sx']
		self.sy = args['sy']
		super().depack(args)


class ListenableDict(dict, Signal):

	def __init__(self):
		Signal.__init__(self)


class MockTime(Signal):
	pass


class MockEngine(Signal):
	eternal = ListenableDict()
	universal = ListenableDict()
	character = ListenableDict()
	time = MockTime()
	closed = False

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.turn = 0
		self._ready = True

	def __setattr__(self, key, value):
		if not hasattr(self, '_ready'):
			super().__setattr__(key, value)
			return
		self.send(self, key=key, value=value)
		super().__setattr__(key, value)

	def next_turn(self, *args, **kwargs):
		self.turn += 1
		kwargs['cb']('next_turn', 'trunk', self.turn, 0, ([], {}))

	def handle(self, *args, **kwargs):
		return {'a': 'b'}

	def commit(self):
		pass


class ELiDEAppTest(GraphicUnitTest):

	def setUp(self):
		super(ELiDEAppTest, self).setUp()
		self.prefix = mkdtemp()
		self.old_argv = sys.argv.copy()
		sys.argv = ['python', '-m', 'ELiDE', self.prefix]
		self.app = ELiDEApp()
		self.app.config = ConfigParser(None)
		self.app.build_config(self.app.config)

	def tearDown(self, fake=False):
		EventLoop.idle()  # let any triggered events fire
		super().tearDown(fake=fake)
		try:
			self.app.stop()
		except RedundantProcessError:
			pass
		shutil.rmtree(self.prefix)
		sys.argv = self.old_argv
