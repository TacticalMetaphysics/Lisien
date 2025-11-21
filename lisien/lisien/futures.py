from __future__ import annotations

import os
import signal
from abc import ABC, abstractmethod
from concurrent.futures import Executor, Future, wait as futwait
from contextlib import contextmanager
from functools import cached_property, wraps
from logging import Logger, LogRecord
from multiprocessing import Pipe
from queue import SimpleQueue, Empty
from threading import Lock, Thread
from time import sleep

from sqlalchemy import Connection

from lisien.proxy.routine import worker_subthread, worker_subprocess
from lisien.proxy.worker_subinterpreter import worker_subinterpreter
from lisien.types import Time, EternalKey, Value, Branch, Turn, Tick
from lisien.util import msgpack_array_header


SUBPROCESS_TIMEOUT = 30
if "LISIEN_SUBPROCESS_TIMEOUT" in os.environ:
	try:
		SUBPROCESS_TIMEOUT = int(os.environ["LISIEN_SUBPROCESS_TIMEOUT"])
	except ValueError:
		SUBPROCESS_TIMEOUT = None
KILL_SUBPROCESS = False
if "LISIEN_KILL_SUBPROCESS" in os.environ:
	KILL_SUBPROCESS = bool(os.environ["LISIEN_KILL_SUBPROCESS"])


class LisienExecutor(Executor, ABC):
	"""Lisien's paralellism

	Starts workers in threads, processes, or interpreters, as needed.

	Usually, you don't want to instantiate these directly -- :class:`Engine`
	will do it for you -- but if you want many :class:`Engine` to share
	the same pool of workers, you can pass the same :class:`LisienExecutor`
	into each.

	"""

	_worker_inputs: list[SimpleQueue] | list[Connection]
	_worker_locks: list[Lock]
	_worker_log_queues: list[SimpleQueue]
	_worker_log_threads: list[Thread]
	_futs_to_start: SimpleQueue[Future]
	workers: int
	logger: Logger

	@abstractmethod
	def __init__(
		self,
		prefix: str | os.PathLike | None,
		logger: Logger,
		time: Time,
		eternal: dict[EternalKey, Value],
		branches: dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]],
		workers: int,
	): ...

	@cached_property
	def lock(self) -> Lock:
		return Lock()

	@cached_property
	def _top_uid(self) -> int:
		return 0

	def get_uid(self):
		ret = self._top_uid
		self._top_uid += 1
		return ret

	def _setup_fut_manager(self, time: Time, workers: int):
		self.workers = workers
		self._worker_updated_btts: list[Time] = [time] * workers
		self._uid_to_fut: dict[int, Future] = {}
		self._futs_to_start = SimpleQueue()
		self._how_many_futs_running = 0
		self._fut_manager_thread = Thread(
			target=self._manage_futs, daemon=True
		)
		self._fut_manager_thread.start()

	def _manage_futs(self):
		if hasattr(self, "_stop_managing_futs"):
			del self._stop_managing_futs
		while not (
			hasattr(self, "_closed") or hasattr(self, "_stop_managing_futs")
		):
			while self._how_many_futs_running < self.workers:
				try:
					fut = self._futs_to_start.get()
					if fut == b"shutdown":
						return
				except Empty:
					break
				if not fut.running() and fut.set_running_or_notify_cancel():
					if hasattr(fut, "_t"):
						fut._t.start()
					else:
						raise RuntimeError("No thread to start", fut)
					self._how_many_futs_running += 1
			sleep(0.001)

	def _sync_log_forever(
		self, logger: Logger, q: SimpleQueue[LogRecord]
	) -> None:
		if hasattr(self, "_stop_sync_log"):
			del self._stop_sync_log
		while not hasattr(self, "_closed") and not hasattr(
			self, "_stop_sync_log"
		):
			recs: list[LogRecord] = []
			while True:
				try:
					rec = q.get()
					if rec == b"shutdown":
						for rec in recs:
							logger.handle(rec)
						return
					recs.append(rec)
				except Empty:
					break
			for rec in recs:
				self.logger.handle(rec)
			sleep(0.5)

	@abstractmethod
	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		if cancel_futures:
			for fut in self._uid_to_fut.values():
				fut.cancel()
		if wait:
			futwait(self._uid_to_fut.values())
		self._uid_to_fut.clear()
		self._stop_managing_futs = True
		self._stop_sync_log = True
		if hasattr(self, "_fut_manager_thread"):
			self._futs_to_start.put(b"shutdown")
			self._fut_manager_thread.join()

	@contextmanager
	def _all_worker_locks_ctx(self):
		for lock in self._worker_locks:
			lock.acquire()
		yield
		for lock in self._worker_locks:
			lock.release()

	@abstractmethod
	def _send_worker_input_bytes(self, i: int, input: bytes) -> None: ...

	@abstractmethod
	def _get_worker_output_bytes(self, i: int) -> bytes: ...

	@staticmethod
	def _all_worker_locks(fn):
		@wraps(fn)
		def call_with_all_worker_locks(self, *args, **kwargs):
			with self._all_worker_locks_ctx():
				return fn(self, *args, **kwargs)

		return call_with_all_worker_locks

	@_all_worker_locks
	def call_every_worker(
		self,
		methodbytes: bytes,
		argbytes: bytes,
		kwargbytes: bytes,
	) -> list[bytes]:
		ret = []
		uids = []
		n = self.workers
		for _ in range(n):
			uid = self.get_uid()
			uids.append(uid)
			uidbytes = uid.to_bytes(8, "little")
			i = uid % n
			self._send_worker_input_bytes(
				i,
				uidbytes
				+ msgpack_array_header(3)
				+ methodbytes
				+ argbytes
				+ kwargbytes,
			)
		for uid in uids:
			i = uid % n
			outbytes: bytes = self._get_worker_output_bytes(i)
			got_uid = int.from_bytes(outbytes[:8], "little")
			assert got_uid == uid
			retbytes = outbytes[8:]
			ret.append(retbytes)
		return ret


class LisienThreadExecutor(LisienExecutor):
	def __init__(
		self,
		prefix: str | os.PathLike | None,
		logger: Logger,
		time: Time,
		eternal_d: dict[EternalKey, Value],
		branches_d: dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]],
		workers: int,
	):
		from queue import SimpleQueue
		from threading import Thread

		self.logger = logger

		self._worker_last_eternal = dict(eternal_d)

		self._worker_threads: list[Thread] = []
		wt = self._worker_threads
		self._worker_inputs: list[SimpleQueue[bytes]] = []
		self._worker_outputs: list[SimpleQueue[bytes]] = []
		wi = self._worker_inputs
		wo = self._worker_outputs
		self._worker_locks: list[Lock] = []
		wlk = self._worker_locks
		self._worker_log_queues: list[SimpleQueue] = []
		wl = self._worker_log_queues
		self._worker_log_threads: list[Thread] = []
		wlt = self._worker_log_threads

		for i in range(workers):
			inq = SimpleQueue()
			outq = SimpleQueue()
			logq = SimpleQueue()
			logthread = Thread(
				target=self._sync_log_forever, args=(logger, logq), daemon=True
			)
			thred = Thread(
				target=worker_subthread,
				name=f"lisien worker {i}",
				args=(
					i,
					prefix,
					dict(branches_d),
					dict(eternal_d),
					inq,
					outq,
					logq,
				),
			)
			wi.append(inq)
			wo.append(outq)
			wl.append(logq)
			wlk.append(Lock())
			wlt.append(logthread)
			wt.append(thred)
			logthread.start()
			thred.start()
			with wlk[-1]:
				inq.put(b"echoImReady")
				if (echoed := outq.get(timeout=5.0)) != b"ImReady":
					raise RuntimeError(
						f"Got garbled output from worker {i}", echoed
					)

		self._setup_fut_manager(time, workers)

	def _send_worker_input_bytes(self, i: int, input: bytes) -> None:
		self._worker_inputs[i].put(input)

	def _get_worker_output_bytes(self, i: int) -> bytes:
		return self._worker_outputs[i].get()

	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		super().shutdown(wait, cancel_futures=cancel_futures)
		for i, (lock, inq, outq, thread, logq, logt) in enumerate(
			zip(
				self._worker_locks,
				self._worker_inputs,
				self._worker_outputs,
				self._worker_threads,
				self._worker_log_queues,
				self._worker_log_threads,
			)
		):
			with lock:
				inq.put(b"shutdown")
				if logt.is_alive():
					logq.put(b"shutdown")
					logt.join()
				thread.join()


class LisienProcessExecutor(LisienExecutor):
	def __init__(
		self,
		prefix: str | os.PathLike | None,
		logger: Logger,
		time: Time,
		eternal: dict[EternalKey, Value],
		branches: dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]],
		workers: int,
	):
		from multiprocessing import get_context
		from multiprocessing.connection import Connection
		from multiprocessing.process import BaseProcess

		self._worker_last_eternal = dict(eternal.items())

		self._worker_processes: list[BaseProcess] = []
		wp = self._worker_processes
		self._mp_ctx = ctx = get_context("spawn")
		self._worker_inputs: list[Connection] = []
		wi = self._worker_inputs
		self._worker_outputs: list[Connection] = []
		wo = self._worker_outputs
		wlk = self._worker_locks = []
		wl = self._worker_log_queues = []
		wlt = self._worker_log_threads = []
		for i in range(workers):
			inpipe_there, inpipe_here = ctx.Pipe(duplex=False)
			outpipe_here, outpipe_there = ctx.Pipe(duplex=False)
			logq = ctx.SimpleQueue()
			logthread = Thread(
				target=self._sync_log_forever, args=(logger, logq), daemon=True
			)
			proc = ctx.Process(
				target=worker_subprocess,
				args=(
					i,
					prefix,
					dict(branches),
					dict(eternal),
					inpipe_there,
					outpipe_there,
					logq,
				),
			)
			wi.append(inpipe_here)
			wo.append(outpipe_here)
			wl.append(logq)
			wlk.append(Lock())
			wlt.append(logthread)
			wp.append(proc)
			logthread.start()
			proc.start()
			with wlk[-1]:
				inpipe_here.send_bytes(b"echoImReady")
				if not outpipe_here.poll(5.0):
					raise TimeoutError(
						f"Couldn't connect to worker process {i} in 5s"
					)
				if (received := outpipe_here.recv_bytes()) != b"ImReady":
					raise RuntimeError(
						f"Got garbled output from worker process {i}", received
					)
		self._setup_fut_manager(time, workers)

	def _send_worker_input_bytes(self, i: int, input: bytes) -> None:
		self._worker_inputs[i].send_bytes(input)

	def _get_worker_output_bytes(self, i: int) -> bytes:
		return self._worker_outputs[i].recv_bytes()

	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		super().shutdown(wait, cancel_futures=cancel_futures)
		for i, (lock, pipein, pipeout, proc, logq, logt) in enumerate(
			zip(
				self._worker_locks,
				self._worker_inputs,
				self._worker_outputs,
				self._worker_processes,
				self._worker_log_queues,
				self._worker_log_threads,
			)
		):
			with lock:
				if proc.is_alive():
					pipein.send_bytes(b"shutdown")
					proc.join(timeout=SUBPROCESS_TIMEOUT)
					if proc.exitcode is None:
						if KILL_SUBPROCESS:
							os.kill(proc.pid, signal.SIGKILL)
						else:
							raise RuntimeError("Worker process didn't exit", i)
					if not KILL_SUBPROCESS and proc.exitcode != 0:
						raise RuntimeError(
							"Worker process didn't exit normally",
							i,
							proc.exitcode,
						)
					proc.close()
				if logt.is_alive():
					logq.put(b"shutdown")
					logt.join(timeout=SUBPROCESS_TIMEOUT)
				pipein.close()
				pipeout.close()
		del self._worker_processes


class LisienInterpreterExecutor(LisienExecutor):
	def __init__(
		self,
		prefix: str | os.PathLike | None,
		logger: Logger,
		time: Time,
		eternal_d: dict[EternalKey, Value],
		branches_d: dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]],
		workers: int,
	) -> None:
		logger.debug(f"starting {workers} worker interpreters")
		from concurrent.interpreters import (
			Interpreter,
			Queue,
			create,
			create_queue,
		)

		self.logger = logger
		self._worker_interpreters: list[Interpreter] = []
		wint = self._worker_interpreters
		self._worker_inputs: list[SimpleQueue] = []
		wi = self._worker_inputs
		self._worker_outputs: list[SimpleQueue] = []
		wo = self._worker_outputs
		self._worker_threads: list[Thread] = []
		wt = self._worker_threads
		self._worker_locks = []
		wlk = self._worker_locks
		self._worker_log_queues = []
		wlq = self._worker_log_queues
		self._worker_log_threads = []
		wlt = self._worker_log_threads
		i = None
		for i in range(workers):
			input = create_queue()
			output = create_queue()
			logq = create_queue()
			terp: Interpreter = create()
			wi.append(input)
			wo.append(output)
			wlq.append(logq)
			lock = Lock()
			wlk.append(lock)
			wint.append(terp)
			logthread = Thread(
				target=self._sync_log_forever, args=(logger, logq), daemon=True
			)
			logthread.start()
			wlt.append(logthread)
			input.put(b"shutdown")
			terp_args = (
				worker_subinterpreter,
				i,
				prefix,
				branches_d,
				eternal_d,
				input,
				output,
				logq,
			)
			terp_kwargs = {
				"function": None,
				"method": None,
				"trigger": None,
				"prereq": None,
				"action": None,
			}
			terp.call(
				*terp_args, **terp_kwargs
			)  # check that we can run the subthread
			if (echoed := output.get(timeout=5.0)) != b"done":
				raise RuntimeError(
					f"Got garbled output from worker terp {i}", echoed
				)
			wt.append(terp.call_in_thread(*terp_args, **terp_kwargs))
			with lock:
				input.put(b"echoImReady")
				if (echoed := output.get(timeout=5.0)) != b"ImReady":
					raise RuntimeError(
						f"Got garbled output from worker terp {i}", echoed
					)
		if not i:
			raise RuntimeError("No workers?", i)
		logger.debug(f"all {i + 1} worker interpreters have started")
		logger.debug(
			"connected function stores to reimporters; setting up fut_manager"
		)
		self._setup_fut_manager(time, workers)
		logger.debug("fut_manager started")

	def _send_worker_input_bytes(self, i: int, input: bytes) -> None:
		self._worker_inputs[i].put(input)

	def _get_worker_output_bytes(self, i: int) -> bytes:
		return self._worker_outputs[i].get()

	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		super().shutdown(wait, cancel_futures=cancel_futures)
		for i, (lock, inq, outq, thread, terp, logq, logt) in enumerate(
			zip(
				self._worker_locks,
				self._worker_inputs,
				self._worker_outputs,
				self._worker_threads,
				self._worker_interpreters,
				self._worker_log_queues,
				self._worker_log_threads,
			)
		):
			with lock:
				inq.put(b"shutdown")
				if logt.is_alive():
					logq.put(b"shutdown")
					logt.join(timeout=SUBPROCESS_TIMEOUT)
				thread.join(timeout=SUBPROCESS_TIMEOUT)
				if terp.is_running():
					terp.close()


class LisienExecutorProxy(LisienExecutor):
	"""A :class:`LisienExecutor` that can itself be shared between processes"""

	executor_class: type[LisienExecutor]

	def __init__(
		self,
		prefix: str | os.PathLike | None,
		logger: Logger,
		time: Time,
		eternal: dict[EternalKey, Value],
		branches: dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]],
		workers: int,
	):
		self._real = self.executor_class(
			prefix, logger, time, eternal, branches, workers
		)
		self._pipe_here, self._pipe_there = Pipe()
		self._listen_thread = Thread(target=self._listen_here, daemon=True)
		self._listen_thread.start()

	def _listen_here(self):
		inst = None
		while inst != "shutdown":
			inst, args, kwargs = self._pipe_here.recv()
			getattr(self, inst)(*args, **kwargs)

	def _listen_there(self):
		inst = None
		while inst != "shutdown":
			inst, args, kwargs = self._pipe_there.recv()
			getattr(self, inst)(*args, **kwargs)

	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		if hasattr(self, "_real"):
			self._real.shutdown(wait, cancel_futures=cancel_futures)
			self._pipe_here.send(
				("shutdown", (wait,), {"cancel_futures": cancel_futures})
			)

	def _send_worker_input_bytes(self, i: int, input: bytes) -> None:
		if hasattr(self, "_real"):
			self._real._send_worker_input_bytes(i, input)
		else:
			self._pipe_there.send(("_send_worker_input_bytes", (i, input), {}))

	def _get_worker_output_bytes(self, i: int) -> bytes:
		if hasattr(self, "_real"):
			return self._real._get_worker_output_bytes(i)
		else:
			self._pipe_there.send(("_get_worker_output_bytes", (i,), {}))
			return self._pipe_there.recv()

	@property
	def workers(self):
		return self._real.workers

	def call_every_worker(
		self,
		methodbytes: bytes,
		argbytes: bytes,
		kwargbytes: bytes,
	) -> list[bytes]:
		if hasattr(self, "_real"):
			return self._real.call_every_worker(
				methodbytes, argbytes, kwargbytes
			)
		self._pipe_there.send(
			(
				"call_every_worker",
				(methodbytes, argbytes, kwargbytes),
				{},
			)
		)
		return self._pipe_there.recv()

	def __getstate__(self):
		return self._pipe_there

	def __setstate__(self, state):
		assert not hasattr(self, "_listen_thread")
		self._pipe_there = state
		self._listen_thread = Thread(target=self._listen_there)
		self._listen_thread.start()


class LisienThreadExecutorProxy(LisienExecutorProxy):
	executor_class = LisienThreadExecutor


class LisienProcessExecutorProxy(LisienExecutorProxy):
	executor_class = LisienProcessExecutor


class LisienInterpreterExecutorProxy(LisienExecutorProxy):
	executor_class = LisienInterpreterExecutor
