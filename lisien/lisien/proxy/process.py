from __future__ import annotations

import sys
import logging
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
	from multiprocessing.connection import Connection
	from queue import Queue

from lisien.exc import OutOfTimelineError
from lisien.proxy import EngineHandle
from lisien.proxy.engine import EngineProxy, WorkerLogHandler, _finish_packing


def worker_subprocess(
	i: int,
	prefix: str,
	branches: dict,
	eternal: dict,
	function: dict,
	method: dict,
	trigger: dict,
	prereq: dict,
	action: dict,
	in_pipe: Union["Connection", "Queue"],
	out_pipe: Union["Connection", "Queue"],
	logq: "Queue",
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
	if hasattr(in_pipe, "recv_bytes"):
		get_input_bytes = in_pipe.recv_bytes
	else:
		get_input_bytes = in_pipe.get
	if hasattr(out_pipe, "send_bytes"):
		send_output_bytes = out_pipe.send_bytes
	else:
		send_output_bytes = out_pipe.put
	while True:
		inst = get_input_bytes()
		if inst == b"shutdown":
			if hasattr(in_pipe, "close"):
				in_pipe.close()
			if logq and hasattr(logq, "close"):
				logq.close()
			send_output_bytes(b"done")
			if hasattr(out_pipe, "close"):
				out_pipe.close()
			return
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


def engine_subprocess(
	args, kwargs, input_pipe, output_pipe, *, log_queue=None
):
	"""Loop to handle one command at a time and pipe results back"""
	engine_handle = None

	def send_output(cmd, r):
		output_pipe.send_bytes(
			engine_handle.pack((cmd, *engine_handle._real.time, r))
		)

	def send_output_bytes(cmd, r):
		output_pipe.send_bytes(
			_finish_packing(
				engine_handle.pack, cmd, *engine_handle._real.time, r
			)
		)

	while True:
		recvd = input_pipe.recv_bytes()
		if recvd == b"shutdown":
			print("shutdown")
			input_pipe.close()
			output_pipe.close()
			if log_queue:
				log_queue.close()
			return 0
		elif recvd.startswith(b"from_archive"):
			if engine_handle is not None:
				engine_handle.close()
			engine_handle = EngineHandle.from_archive(
				recvd.removeprefix(b"from_archive")
			)
			send_output("get_btt", engine_handle.get_btt())
			continue
		elif engine_handle is None:
			engine_handle = EngineHandle(*args, log_queue=log_queue, **kwargs)
			send_output("get_btt", engine_handle.get_btt())
			continue
		instruction = engine_handle.unpack(recvd)
		_engine_subroutine_step(
			engine_handle, instruction, send_output, send_output_bytes
		)


def _engine_subroutine_step(
	handle: EngineHandle, instruction: dict, send_output, send_output_bytes
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
		send_output_bytes(cmd, r)
	else:
		send_output(cmd, r)
	if hasattr(handle, "_after_ret"):
		handle._after_ret()
		del handle._after_ret


def engine_subthread(args, kwargs, input_queue, output_queue):
	def send_output(cmd, r):
		output_queue.put(
			engine_handle.pack((cmd, *engine_handle._real.time, r))
		)

	def send_output_bytes(cmd, r):
		output_queue.put(
			_finish_packing(
				engine_handle.pack, cmd, *engine_handle._real.time, r
			)
		)

	engine_handle = None

	while True:
		instruction = input_queue.get()
		if instruction == "shutdown" or instruction == b"shutdown":
			return
		if (
			isinstance(instruction, tuple)
			and instruction
			and instruction[0] == "from_archive"
		):
			engine_handle = EngineHandle.from_archive(instruction[1])
			send_output("get_btt", engine_handle.get_btt())
			continue
		if engine_handle is None:
			engine_handle = EngineHandle(*args, **kwargs)
			send_output("get_btt", engine_handle.get_btt())
			continue
		_engine_subroutine_step(
			engine_handle,
			engine_handle.unpack(instruction),
			send_output,
			send_output_bytes,
		)
