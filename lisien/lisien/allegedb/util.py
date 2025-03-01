import gc
from contextlib import contextmanager
from functools import wraps


def garbage(arg: callable = None, collect=True):
	"""Disable the garbage collector, then re-enable it when done.

	May be used as a context manager or a decorator.

	:param collect: Whether to immediately run a collection after re-enabling
		the garbage collector. Default ``True``.

	"""

	@contextmanager
	def garbage_ctx(collect=True):
		"""Context manager to disable the garbage collector

		:param collect: Whether to immediately collect garbage upon context exit

		"""
		gc_was_active = gc.isenabled()
		if gc_was_active:
			gc.disable()
		yield
		if gc_was_active:
			gc.enable()
		if collect:
			gc.collect()

	def garbage_dec(fn: callable, collect=True) -> callable:
		"""Decorator to disable the garbage collector for a function

		:param collect: Whether to immediately collect garbage when the function returns

		"""

        @wraps(fn)
		def garbage(*args, **kwargs):
			with garbage_ctx(collect=collect):
				return fn(*args, **kwargs)

		return garbage

	if arg is None:
		return garbage_ctx(collect=collect)
	else:
		return garbage_dec(arg, collect=collect)


def world_locked(fn: callable) -> callable:
	"""Decorator for functions that alter the world state

	They will hold a reentrant lock, preventing more than one function
	from mutating the world at a time.

	"""

	@wraps(fn)
	def lockedy(*args, **kwargs):
		with args[0].world_lock:
			return fn(*args, **kwargs)

	return lockedy
