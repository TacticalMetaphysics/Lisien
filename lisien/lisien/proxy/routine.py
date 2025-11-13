# This file is part of Lisien, a framework for life simulation games.
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
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
	from multiprocessing.connection import Connection
	from queue import Queue

from lisien.exc import OutOfTimelineError
from lisien.proxy import EngineHandle
from lisien.proxy.engine import EngineProxy, WorkerLogHandler, _finish_packing


def worker_subroutine(
	i: int,
	prefix: str,
	branches: dict,
	eternal: dict,
	get_input_bytes: Callable[[], bytes],
	send_output_bytes: Callable[[bytes], None],
	logq: Queue,
	*,
	function: dict | None = None,
	method: dict | None = None,
	trigger: dict | None = None,
	prereq: dict | None = None,
	action: dict | None = None,
):
	logger = logging.getLogger(f"lisien worker {i}")
	handler = WorkerLogHandler(logq, logging.DEBUG, i)
	logger.addHandler(handler)
	eng = EngineProxy(
		None,
		None,
		logger,
		prefix=prefix,
		worker_index=i,
		eternal=eternal,
		branches=branches,
		function=function,
		method=method,
		trigger=trigger,
		prereq=prereq,
		action=action,
	)
	pack = eng.pack
	unpack = eng.unpack
	eng._initialized = False

	while True:
		inst = get_input_bytes()
		if inst == b"shutdown":
			if logq and hasattr(logq, "close"):
				logq.close()
			send_output_bytes(b"done")
			return 0
		if inst.startswith(b"echo"):
			send_output_bytes(inst.removeprefix(b"echo"))
			continue
		uid = int.from_bytes(inst[:8], "little")
		(method, args, kwargs) = unpack(inst[8:])
		if isinstance(method, str):
			method = getattr(eng, method)
		try:
			ret = method(*args, **kwargs)
		except Exception as ex:
			ret = ex
			if uid == sys.maxsize:
				msg = repr(ex)
				logq.put((50, msg))
				import traceback

				traceback.print_exc(file=sys.stderr)
				sys.exit(msg)
		if uid != sys.maxsize:
			send_output_bytes(inst[:8] + pack(ret))
		eng._initialized = True


def worker_subprocess(
	i: int,
	prefix: str,
	branches: dict,
	eternal: dict,
	in_pipe: Connection,
	out_pipe: Connection,
	logq: Queue,
	*,
	function: dict | None = None,
	method: dict | None = None,
	trigger: dict | None = None,
	prereq: dict | None = None,
	action: dict | None = None,
):
	return worker_subroutine(
		i,
		prefix,
		branches,
		eternal,
		in_pipe.recv_bytes,
		out_pipe.send_bytes,
		logq,
		function=function,
		method=method,
		trigger=trigger,
		prereq=prereq,
		action=action,
	)


def worker_subthread(
	i: int,
	prefix: str,
	branches: dict,
	eternal: dict,
	in_queue: Queue,
	out_queue: Queue,
	logq: Queue,
	*,
	function: dict | None = None,
	method: dict | None = None,
	trigger: dict | None = None,
	prereq: dict | None = None,
	action: dict | None = None,
):
	return worker_subroutine(
		i,
		prefix,
		branches,
		eternal,
		in_queue.get,
		out_queue.put,
		logq,
		function=function,
		method=method,
		trigger=trigger,
		prereq=prereq,
		action=action,
	)


def _engine_subroutine_step(
	handle: EngineHandle, instruction: dict, send_output, send_output_prepacked
):
	silent = instruction.pop("silent", False)
	cmd = instruction.pop("command")
	branching = instruction.pop("branching", False)
	r = None
	try:
		if branching:
			try:
				r = getattr(handle, cmd)(**instruction)
			except OutOfTimelineError:
				handle.increment_branch()
				r = getattr(handle, cmd)(**instruction)
		else:
			r = getattr(handle, cmd)(**instruction)
	except AssertionError:
		raise
	except Exception as ex:
		handle._real.logger.error(repr(ex))
		send_output(cmd, ex)
		return
	if silent:
		return
	if hasattr(getattr(handle, cmd), "prepacked"):
		send_output_prepacked(cmd, r)
	else:
		send_output(cmd, r)
	if hasattr(handle, "_after_ret"):
		handle._after_ret()
		del handle._after_ret


def engine_subroutine(
	args,
	kwargs,
	get_input_bytes: Callable[[], bytes],
	send_output_bytes: Callable[[bytes], None],
	log_queue=None,
):
	engine_handle = None

	def send_output(cmd, r):
		send_output_bytes(
			engine_handle.pack((cmd, *engine_handle._real.time, r))
		)

	def send_output_prepacked(cmd, r):
		send_output_bytes(
			_finish_packing(
				engine_handle.pack, cmd, *engine_handle._real.time, r
			)
		)

	while True:
		recvd = get_input_bytes()
		if recvd == b"shutdown":
			return 0
		if recvd.startswith(b"echo"):
			send_output_bytes(recvd.removeprefix(b"echo"))
			continue
		if recvd.startswith(b"from_archive"):
			if engine_handle is not None:
				engine_handle.close()
			engine_handle = EngineHandle.from_archive(
				recvd.removeprefix(b"from_archive"), log_queue=log_queue
			)
			send_output("get_btt", engine_handle.get_btt())
			continue
		if engine_handle is None:
			engine_handle = EngineHandle(*args, log_queue=log_queue, **kwargs)
			send_output("get_btt", engine_handle.get_btt())
			continue
		_engine_subroutine_step(
			engine_handle,
			engine_handle.unpack(recvd),
			send_output,
			send_output_prepacked,
		)


def engine_subprocess(args, kwargs, in_pipe, out_pipe, log_queue):
	return engine_subroutine(
		args, kwargs, in_pipe.recv_bytes, out_pipe.send_bytes, log_queue
	)


def engine_subthread(args, kwargs, input_queue, output_queue, log_queue):
	return engine_subroutine(
		args, kwargs, input_queue.get, output_queue.put, log_queue
	)
