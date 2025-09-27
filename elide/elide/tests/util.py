from functools import partial
from typing import Callable

from blinker import Signal
from kivy.app import App
from kivy.base import EventLoop


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
	return (
		all_spots_placed(board, char)
		and all_pawns_placed(board, char)
		and all_arrows_placed(board, char)
	)


def idle_until(
	condition: Callable[[], bool] | None = None,
	timeout: int | None = 100,
	message: str | None = None,
) -> partial | None:
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
	if message is None:
		if hasattr(condition, "__name__"):
			message = f"{condition.__name__} timed out"
		else:
			message = "Timed out"
	raise TimeoutError(message)


idle100 = partial(idle_until, timeout=100)


class ListenableDict(dict, Signal):
	def __init__(self):
		Signal.__init__(self)


def advance_frames(frames: int) -> None:
	from kivy.base import EventLoop

	for _ in range(frames):
		EventLoop.idle()


def transition_over():
	# Even though there's "no transition," the screen manager still considers
	# a transition to be active for a few frames, and will refuse input for
	# the duration
	return not App.get_running_app().manager.transition.is_active
