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
"""Executor classes, for Lisien's various forms of parallelism

Lisien's workers are stateful--they each hold a shallow copy of the world state.
These executors keep that copy synchronized with the current state of the game.

"""

from __future__ import annotations

import os
import sys
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Executor, Future
from concurrent.futures import wait as futwait
from contextlib import contextmanager
from functools import cached_property, partial, wraps
from logging import Logger, LogRecord
from multiprocessing import SimpleQueue as MPSimpleQueue
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.context import DefaultContext, ForkContext, SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
from threading import Lock, Thread
from time import sleep
from typing import TYPE_CHECKING, Callable, Literal

from attrs import Factory, define, field

from .proxy.routine import worker_subprocess, worker_subthread
from .proxy.worker_subinterpreter import worker_subinterpreter
from .types import AbstractEngine, Branch, EternalKey, Tick, Time, Turn, Value
from .util import msgpack_array_header, msgpack_map_header, unpack_expected

if TYPE_CHECKING:
	try:
		from concurrent.interpreters import Interpreter
	except ModuleNotFoundError:
		Interpreter = type(object)


SUBPROCESS_TIMEOUT = 30
if "LISIEN_SUBPROCESS_TIMEOUT" in os.environ:
	try:
		SUBPROCESS_TIMEOUT = int(os.environ["LISIEN_SUBPROCESS_TIMEOUT"])
	except ValueError:
		SUBPROCESS_TIMEOUT = None
KILL_SUBPROCESS = False
if "LISIEN_KILL_SUBPROCESS" in os.environ:
	KILL_SUBPROCESS = bool(os.environ["LISIEN_KILL_SUBPROCESS"])


@define
class Worker(ABC):
	"""Abstract class representing one worker in an :class:`Executor`

	Implementations may represent a worker thread in the same interpreter,
	a worker thread in a subinterpreter, or a worker process.

	"""

	last_update: Time
	unpack: Callable[[bytes], Value]
	lock: Lock = field(init=False, factory=Lock)

	@abstractmethod
	def send_input_bytes(self, input_bytes: bytes) -> None:
		"""Send packed data to the worker"""

	@abstractmethod
	def get_output_bytes(self) -> bytes:
		"""Get a reply from the worker"""

	@abstractmethod
	def shutdown(self) -> None:
		"""Stop running the worker"""

	@classmethod
	@abstractmethod
	def from_executor(cls, executor: Executor) -> Worker:
		"""The usual way of instantiating a :class:`Worker`"""

	@staticmethod
	def _sync_log_forever(
		logger: Logger, q: Queue[LogRecord | Literal[b"shutdown"]]
	) -> None:
		while (got := q.get()) != b"shutdown":
			logger.handle(got)


@define
class _BaseLisienExecutor[WRKR: Worker](Executor, ABC):
	@property
	def prefix(self) -> Path | None:
		return self.engine._prefix

	@property
	def logger(self) -> Logger:
		return self.engine.logger

	def log(self, *args):
		self.logger.log(*args)

	def debug(self, *args):
		self.logger.debug(*args)

	def info(self, *args):
		self.logger.info(*args)

	def warning(self, *args):
		self.logger.warning(*args)

	def error(self, *args):
		self.logger.error(*args)

	def critical(self, *args):
		self.logger.critical(*args)

	@property
	def time(self) -> Time:
		return self.engine.branch, self.engine.turn, self.engine.tick

	@property
	def branch(self) -> Branch:
		return self.engine.branch

	@property
	def turn(self) -> Turn:
		return self.engine.turn

	@property
	def tick(self) -> Tick:
		return self.engine.tick

	@property
	def eternal(self) -> dict[EternalKey, Value]:
		return dict(self.engine.eternal)

	@property
	def branches(
		self,
	) -> dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]]:
		return dict(self.engine._branches_d)

	@property
	def workers(self) -> int:
		return self.engine.workers

	@property
	def random_seed(self) -> int:
		return self.engine.random_seed

	def pack(self, obj: Value) -> bytes:
		return self.engine.pack(obj)

	def unpack(self, b: bytes) -> Value:
		return self.engine.unpack(b)

	@cached_property
	def _workers(self) -> list[WRKR]:
		return []


@define
class Executor[WRKR: Worker](_BaseLisienExecutor[WRKR], ABC):
	"""Abstract class for Lisien's parallel execution

	Starts workers in threads, processes, or interpreters, depending on which
	subclass you use.

	Usually, you don't want to instantiate these directly -- :class:`Engine`
	will do it for you -- but if you want to close an :class:`Engine` while
	keeping its workers alive, and reuse them when next you start the game,
	you can do that by creating your own :class:`Executor`, passing it to
	:class:`lisien.Engine`, and holding onto it until the one engine's shut
	down, and it's time to start the next.

	These are stateful, and can only serve one :class:`Engine` at a time.

	"""

	engine: AbstractEngine = field()

	lock: Lock = field(init=False, factory=Lock)
	_top_uid: int = field(init=False, default=0)
	_uid_to_fut: dict[int, Future] = field(init=False, factory=dict)
	_worker_last_eternal: dict[EternalKey, Value] = field(
		init=False, factory=dict
	)
	_worker_last_branches: dict[
		Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]
	] = field(init=False, factory=dict)
	_stop_managing_futs: bool = False
	_futs_to_start: SimpleQueue[Future | Literal[b"shutdown"]] = field(
		init=False, factory=SimpleQueue
	)
	_how_many_futs_running: int = field(init=False, default=0)

	def _make_fut_manager_thread(self):
		return Thread(target=self._manage_futs)

	_fut_manager_thread: Thread = field(
		init=False, default=Factory(_make_fut_manager_thread, takes_self=True)
	)

	def _call_in_worker(
		self,
		uid,
		method,
		future: Future,
		*args,
		update=True,
		**kwargs,
	):
		i = uid % len(self._workers)
		uidbytes = uid.to_bytes(8, "little")
		argbytes = self.pack((method, args, kwargs))
		info = self._workers[i]
		with info.lock:
			if update:
				self._update_worker_process_state(i, lock=False)
			if hasattr(info, "input_queue"):
				info.input_queue.put(uidbytes + argbytes)
				output_bytes: bytes = info.output_queue.get()
			else:
				info.input_connection.send_bytes(uidbytes + argbytes)
				output_bytes: bytes = info.output_connection.recv_bytes()
		got_uid = int.from_bytes(output_bytes[:8], "little")
		result = self.unpack(output_bytes[8:])
		assert got_uid == uid
		self._how_many_futs_running -= 1
		del self._uid_to_fut[uid]
		if isinstance(result, Exception):
			future.set_exception(result)
		else:
			future.set_result(result)

	def _update_worker_process_state(self, i, lock=True):
		info = self._workers[i]
		branch_from, turn_from, tick_from = info.last_update
		if (branch_from, turn_from, tick_from) == self.time:
			return
		old_eternal = self._worker_last_eternal
		new_eternal = self._worker_last_eternal = dict(self.eternal.items())
		eternal_delta = {
			k: new_eternal.get(k, ...)
			for k in old_eternal.keys() | new_eternal.keys()
			if old_eternal.get(k, ...) != new_eternal.get(k, ...)
		}
		if branch_from == self.branch:
			delt = self.engine._get_branch_delta(
				branch_from, turn_from, tick_from, self.turn, self.tick
			)
			delt["eternal"] = eternal_delta
			argbytes = sys.maxsize.to_bytes(8, "little") + self.pack(
				(
					"_upd",
					(
						None,
						self.branch,
						self.turn,
						self.tick,
						(None, delt),
					),
					{},
				)
			)
		else:
			argbytes = self.engine._get_worker_kf_payload()
		info = self._workers[i]
		if hasattr(info, "input_connection"):
			put = info.input_connection.send_bytes
		else:
			put = info.input_queue.put
		if lock:
			with info.lock:
				put(argbytes)
		else:
			put(argbytes)
		info.last_update = self.time
		self.debug(f"Updated worker {i} at {info.last_update}")

	def get_uid(self):
		ret = self._top_uid
		self._top_uid += 1
		return ret

	def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
		ret = Future()
		uid = self.get_uid()
		ret._t = Thread(
			target=self._call_in_worker,
			args=(uid, fn, ret, *args),
			kwargs=kwargs,
		)
		ret.uid = uid
		self._uid_to_fut[uid] = ret
		self._futs_to_start.put(ret)
		return ret

	def _setup_workers(self) -> None:
		self._worker_last_branches.clear()
		self._worker_last_branches.update(self.engine._branches_d)
		self._worker_last_eternal.clear()
		self._worker_last_eternal.update(self.eternal)
		while len(self._workers) < self.engine.workers:
			self._workers.append(self._make_worker())
		while len(self._workers) > self.engine.workers:
			self._workers.pop().shutdown()

	@abstractmethod
	def _make_worker(self) -> WRKR: ...

	def _manage_futs(self):
		self._stop_managing_futs = False
		while not self._stop_managing_futs:
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

	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		if cancel_futures:
			for fut in self._uid_to_fut.values():
				fut.cancel()
		if wait:
			futwait(self._uid_to_fut.values())
		for fut in self._uid_to_fut.values():
			fut._t.join()
		self._uid_to_fut.clear()
		self._stop_managing_futs = True
		self._stop_sync_log = True
		if self._fut_manager_thread.is_alive():
			self._futs_to_start.put(b"shutdown")
			self._fut_manager_thread.join()
			self._fut_manager_thread = self._make_fut_manager_thread()
		while self._workers:
			self._workers.pop().shutdown()

	@contextmanager
	def all_worker_locks_ctx(self):
		for worker in self._workers:
			worker.lock.acquire()
		yield
		for worker in self._workers:
			worker.lock.release()

	def _send_worker_input_bytes(self, i: int, input_bytes: bytes) -> None:
		self._workers[i].send_input_bytes(input_bytes)

	def _get_worker_output_bytes(self, i: int) -> bytes:
		return self._workers[i].get_output_bytes()

	@staticmethod
	def _all_worker_locks_dec(fn):
		@wraps(fn)
		def call_with_all_worker_locks(self, *args, **kwargs):
			with self.all_worker_locks_ctx():
				return fn(self, *args, **kwargs)

		return call_with_all_worker_locks

	@_all_worker_locks_dec
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

	def restart(self):
		"""Overwrite all workers' state to match my attributes

		For when the Lisien engine is done working with this executor,
		and you want to reuse it for another Lisien engine, likely in
		a unit test.

		"""
		if self._fut_manager_thread.is_alive():
			self._futs_to_start.put(b"shutdown")
			self._fut_manager_thread.join()
			self._fut_manager_thread = self._make_fut_manager_thread()
		initial_payload = self.engine._get_worker_kf_payload()
		self._setup_workers()
		assert len(self._workers) == self.workers

		from umsgpack import packb

		self.call_every_worker(
			packb("_restart"),
			packb(
				(
					str(self.prefix),
					self.time,
					self.eternal,
					self.branches,
					self.random_seed,
				)
			),
			msgpack_map_header(0),
		)
		for i in range(self.workers):
			self._send_worker_input_bytes(i, initial_payload)
		try:
			self._fut_manager_thread.start()
		except RuntimeError:
			self._fut_manager_thread = self._make_fut_manager_thread()
			self._fut_manager_thread.start()


@define
class ThreadWorker(Worker):
	last_update: Time
	unpack: Callable[[bytes], Value]
	input_queue: SimpleQueue[bytes]
	output_queue: SimpleQueue[bytes]
	worker_thread: Thread
	log_queue: SimpleQueue[LogRecord | Literal[b"shutdown"]]
	log_thread: Thread
	lock: Lock = field(init=False, factory=Lock)

	def send_input_bytes(self, input_bytes: bytes) -> None:
		self.input_queue.put(input_bytes)

	def get_output_bytes(self) -> bytes:
		return self.output_queue.get()

	def shutdown(self) -> None:
		with self.lock:
			self.input_queue.put(b"shutdown")
			try:
				got = self.output_queue.get(timeout=15.0)
			except Empty:
				raise TimeoutError("No response to worker shutdown")
			unpack_expected(self.unpack, got, b"done")
			self.worker_thread.join(15.0)
			if self.worker_thread.is_alive():
				raise TimeoutError(
					"Worker thread didn't terminate", self.worker_thread
				)
			if self.log_thread.is_alive():
				self.log_queue.put(b"shutdown")
				self.log_thread.join()

	@classmethod
	def from_executor(cls, executor: Executor) -> ThreadWorker:
		engine = executor.engine
		i = len(executor._workers)
		inq = SimpleQueue()
		outq = SimpleQueue()
		logq = SimpleQueue()
		logthread = Thread(
			target=cls._sync_log_forever, args=(executor.logger, logq)
		)
		thred = Thread(
			target=worker_subthread,
			name=f"lisien worker {i}",
			args=(
				i,
				engine._prefix,
				executor._worker_last_branches,
				executor._worker_last_eternal,
				engine.random_seed,
				inq,
				outq,
				logq,
			),
		)
		logthread.start()
		thred.start()
		inq.put(b"echoImReady")
		try:
			echoed = outq.get(timeout=10.0)
		except Empty:
			raise TimeoutError("No echo from worker thread")
		unpack_expected(engine.unpack, echoed, b"ImReady")
		return cls(
			(engine.branch, engine.turn, engine.tick),
			engine.unpack,
			inq,
			outq,
			thred,
			logq,
			logthread,
		)


@define
class ThreadExecutor(Executor[ThreadWorker]):
	def _make_worker(self) -> ThreadWorker:
		return ThreadWorker.from_executor(self)


@define
class ProcessWorker(Worker):
	last_update: Time
	unpack: Callable[[bytes], Value]
	process: BaseProcess
	input_connection: Connection
	output_connection: Connection
	log_queue: MPSimpleQueue[LogRecord | Literal[b"shutdown"]]
	log_thread: Thread

	def send_input_bytes(self, input_bytes: bytes) -> None:
		self.input_connection.send_bytes(input_bytes)

	def get_output_bytes(self) -> bytes:
		return self.output_connection.recv_bytes()

	def shutdown(self) -> None:
		with self.lock:
			self.input_connection.send_bytes(b"shutdown")
			if not self.output_connection.poll(10):
				raise TimeoutError("No response to shutdown")
			unpack_expected(
				self.unpack, self.output_connection.recv_bytes(), b"done"
			)
			if self.log_thread.is_alive():
				self.log_queue.put(b"shutdown")
				self.log_thread.join(timeout=10.0)
			self.process.join(timeout=10.0)
			if self.process.is_alive():
				self.process.kill()
				self.process.join(timeout=10.0)
				if self.process.is_alive():
					self.process.terminate()
			self.process.close()

	@classmethod
	def from_executor(cls, executor: Executor) -> ProcessWorker:
		i = len(executor._workers)
		engine = executor.engine
		ctx = executor._mp_ctx
		lock = Lock()
		with lock:
			inpipe_there, inpipe_here = ctx.Pipe(duplex=False)
			outpipe_here, outpipe_there = ctx.Pipe(duplex=False)
			logq = ctx.SimpleQueue()
			logthread = Thread(
				target=cls._sync_log_forever, args=(executor.logger, logq)
			)
			proc = ctx.Process(
				target=worker_subprocess,
				args=(
					i,
					engine._prefix,
					executor._worker_last_branches,
					executor._worker_last_eternal,
					engine.random_seed,
					inpipe_there,
					outpipe_there,
					logq,
				),
			)
			logthread.start()
			proc.start()
			inpipe_here.send_bytes(b"echoImReady")
			if not outpipe_here.poll(15.0):
				raise TimeoutError(
					f"Couldn't connect to worker process {i} in 5s"
				)
			unpack_expected(
				engine.unpack, outpipe_here.recv_bytes(), b"ImReady"
			)
			return cls(
				(engine.branch, engine.turn, engine.tick),
				engine.unpack,
				proc,
				inpipe_here,
				outpipe_here,
				logq,
				logthread,
			)


@define
class ProcessExecutor(Executor[ProcessWorker]):
	_mp_ctx: ForkContext | SpawnContext | DefaultContext = field(
		init=False, factory=partial(get_context, "spawn")
	)

	def _make_worker(self) -> ProcessWorker:
		return ProcessWorker.from_executor(self)


@define
class InterpreterWorker(Worker):
	last_update: Time
	unpack: Callable[[bytes], Value]
	interpreter: Interpreter
	thread: Thread
	input_queue: Queue[bytes]
	output_queue: Queue[bytes]
	log_queue: Queue[LogRecord | Literal[b"shutdown"]]
	log_thread: Thread

	def send_input_bytes(self, input_bytes: bytes) -> None:
		self.input_queue.put(input_bytes)

	def get_output_bytes(self) -> bytes:
		return self.output_queue.get()

	def shutdown(self) -> None:
		with self.lock:
			self.input_queue.put(b"shutdown")
			try:
				echoed = self.output_queue.get(timeout=10)
			except Empty:
				raise TimeoutError("Worker terp didn't shut down", echoed)
			unpack_expected(self.unpack, echoed, b"done")
			if self.thread.is_alive():
				self.thread.join(timeout=5)
				if self.thread.is_alive():
					raise TimeoutError("Worker thread didn't join")
			if self.log_thread.is_alive():
				self.log_queue.put(b"shutdown")
				self.log_thread.join(timeout=5)
				if self.log_thread.is_alive():
					raise TimeoutError("Worker log thread didn't join")
			if self.interpreter.is_running():
				self.interpreter.close()

	@classmethod
	def from_executor(
		cls,
		executor: InterpreterExecutor,
	) -> InterpreterWorker:
		from concurrent.interpreters import create, create_queue

		engine = executor.engine
		i = len(executor._workers)
		input = create_queue()
		output = create_queue()
		logq = create_queue()
		terp = create()
		logthread = Thread(
			target=cls._sync_log_forever, args=(executor.logger, logq)
		)
		logthread.start()
		input.put(b"shutdown")
		terp_args = (
			worker_subinterpreter,
			i,
			engine._prefix,
			executor._worker_last_branches,
			executor._worker_last_eternal,
			engine.random_seed,
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
		if (echoed := output.get(timeout=5)) != b"done":
			raise RuntimeError(
				f"Got garbled output from worker terp {i}", echoed
			)
		workthread = terp.call_in_thread(*terp_args, **terp_kwargs)
		input.put(b"echoImReady")
		if (echoed := output.get(timeout=5)) != b"ImReady":
			raise RuntimeError(
				f"Got garbled output from worker terp {i}", echoed
			)
		return cls(
			(engine.branch, engine.turn, engine.tick),
			engine.unpack,
			terp,
			workthread,
			input,
			output,
			logq,
			logthread,
		)


@define
class InterpreterExecutor(Executor[InterpreterWorker]):
	def _make_worker(self) -> InterpreterWorker:
		return InterpreterWorker.from_executor(self)
