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

import os
import sys
import signal
from abc import ABC, abstractmethod
from concurrent.futures import Executor, Future, wait as futwait
from contextlib import contextmanager
from functools import cached_property, wraps, partial
from logging import Logger, LogRecord
from multiprocessing import Pipe
from pathlib import Path
from queue import SimpleQueue, Empty
from threading import Lock, Thread
from time import sleep
from typing import ClassVar, Callable

from attrs import Factory, define, field
from sqlalchemy import Connection

from .proxy.routine import worker_subthread, worker_subprocess
from .proxy.worker_subinterpreter import worker_subinterpreter
from .types import AbstractEngine, Time, EternalKey, Value, Branch, Turn, Tick
from .util import msgpack_array_header, msgpack_map_header


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
class _BaseLisienExecutor(Executor, ABC):
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
	def _worker_updated_btts(self):
		return [self.time] * self.workers


@define
class LisienExecutor(_BaseLisienExecutor, ABC):
	"""Lisien's parallelism

	Starts workers in threads, processes, or interpreters, as needed.

	Usually, you don't want to instantiate these directly -- :class:`Engine`
	will do it for you -- but if you want many :class:`Engine` to share
	the same pool of workers, you can pass the same :class:`LisienExecutor`
	into each.

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
	_futs_to_start: SimpleQueue[Future] = field(
		init=False, factory=SimpleQueue
	)
	_how_many_futs_running: int = field(init=False, default=0)

	def _make_fut_manager_thread(self):
		return Thread(target=self._manage_futs, daemon=True)

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
		i = uid % len(self._worker_inputs)
		uidbytes = uid.to_bytes(8, "little")
		argbytes = self.pack((method, args, kwargs))
		with self._worker_locks[i]:
			if update:
				self._update_worker_process_state(i, lock=False)
			input = self._worker_inputs[i]
			output = self._worker_outputs[i]
			if hasattr(input, "send_bytes"):
				input.send_bytes(uidbytes + argbytes)
			else:
				input.put(uidbytes + argbytes)
			if hasattr(output, "recv_bytes"):
				output_bytes: bytes = output.recv_bytes()
			else:
				output_bytes: bytes = output.get()
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
		branch_from, turn_from, tick_from = self._worker_updated_btts[i]
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
		input = self._worker_inputs[i]
		if hasattr(input, "send_bytes"):
			put = input.send_bytes
		else:
			put = input.put
		if lock:
			with self._worker_locks[i]:
				put(argbytes)
				self._worker_updated_btts[i] = tuple(self.time)
		else:
			put(argbytes)
			self._worker_updated_btts[i] = tuple(self.time)
		self.debug(f"Updated worker {i} at {self._worker_updated_btts[i]}")

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

	@abstractmethod
	def _setup_workers(self, engine) -> None:
		self._worker_last_branches.clear()
		self._worker_last_branches.update(engine._branches_d)
		self._worker_last_eternal.clear()
		self._worker_last_eternal.update(self.eternal)
		self._worker_updated_btts.clear()
		self._worker_updated_btts.extend(
			[tuple(self.engine.time)] * self.workers
		)

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
		for fut in self._uid_to_fut.values():
			fut._t.join()
		self._uid_to_fut.clear()
		self._stop_managing_futs = True
		self._stop_sync_log = True
		if self._fut_manager_thread.is_alive():
			self._futs_to_start.put(b"shutdown")
			self._fut_manager_thread.join()
			self._fut_manager_thread = self._make_fut_manager_thread()

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

	def restart(self, keyframe_cb: Callable[[], bytes] | None = None):
		"""Overwrite all workers' state to match my attributes

		For when the Lisien engine is done working with this executor,
		and you want to reuse it for another Lisien engine, likely in
		a unit test.

		:param keyframe_cb: Function to get the initial payload. If omitted,
		the workers will have only the state expressed in the engine's attributes.
		Should be the :meth:`_get_worker_kf_payload` method of
		:class:`lisien.Engine`.

		"""
		if self._fut_manager_thread.is_alive():
			self._futs_to_start.put(b"shutdown")
			self._fut_manager_thread.join()
			self._fut_manager_thread = self._make_fut_manager_thread()
		self._setup_workers(self.engine)
		assert len(self._worker_inputs) == self.workers

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
		if keyframe_cb:
			initial_payload = keyframe_cb()
			for i in range(self.workers):
				self._send_worker_input_bytes(i, initial_payload)
		try:
			self._fut_manager_thread.start()
		except RuntimeError:
			self._fut_manager_thread = self._make_fut_manager_thread()
			self._fut_manager_thread.start()


@define
class LisienThreadExecutor(LisienExecutor):
	_worker_inputs: list[SimpleQueue[bytes]] = field(init=False, factory=list)
	_worker_outputs: list[SimpleQueue[bytes]] = field(init=False, factory=list)
	_worker_threads: list[Thread] = field(init=False, factory=list)
	_worker_locks: list[Lock] = field(init=False, factory=list)
	_worker_log_threads: list[Thread] = field(init=False, factory=list)
	_worker_log_queues: list[SimpleQueue[LogRecord]] = field(
		init=False, factory=list
	)

	def _setup_workers(self, engine):
		super()._setup_workers(engine)

		self._stop_sync_log = True
		wi = self._worker_inputs
		wt = self._worker_threads
		wlk = self._worker_locks
		wlt = self._worker_log_threads
		wl = self._worker_log_queues
		wo = self._worker_outputs
		assert len(wi) == len(wt) == len(wlk) == len(wlt) == len(wl) == len(wo)
		for i in range(self.workers - len(self._worker_inputs)):
			inq = SimpleQueue()
			outq = SimpleQueue()
			logq = SimpleQueue()
			logthread = Thread(
				target=self._sync_log_forever,
				args=(self.logger, logq),
				daemon=True,
			)
			thred = Thread(
				target=worker_subthread,
				name=f"lisien worker {i}",
				args=(
					i,
					engine._prefix,
					self._worker_last_branches,
					self._worker_last_eternal,
					engine.random_seed,
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
		for i in range(self.workers):
			inq = wi[i]
			outq = wo[i]
			with wlk[-1]:
				inq.put(b"echoImReady")
				if (echoed := outq.get(timeout=5.0)) != b"ImReady":
					raise RuntimeError(
						f"Got garbled output from worker {i}", echoed
					)

	def _send_worker_input_bytes(self, i: int, input: bytes) -> None:
		self._worker_inputs[i].put(input)

	def _get_worker_output_bytes(self, i: int) -> bytes:
		return self._worker_outputs[i].get()

	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		super().shutdown(wait, cancel_futures=cancel_futures)
		todo = (
			self._worker_locks,
			self._worker_inputs,
			self._worker_outputs,
			self._worker_threads,
			self._worker_log_queues,
			self._worker_log_threads,
		)
		while any(todo):
			(lock, inq, outq, thread, logq, logt) = (
				mylist.pop() for mylist in todo
			)
			with lock:
				inq.put(b"shutdown")
				if logt.is_alive():
					logq.put(b"shutdown")
					logt.join()
				thread.join()


@define
class LisienProcessExecutor(LisienExecutor):
	from multiprocessing import SimpleQueue, get_context
	from multiprocessing.process import BaseProcess
	from multiprocessing.context import (
		ForkContext,
		SpawnContext,
		DefaultContext,
	)
	from multiprocessing.connection import Connection

	_worker_processes: list[BaseProcess] = field(init=False, factory=list)
	_mp_ctx: ForkContext | SpawnContext | DefaultContext = field(
		init=False, factory=partial(get_context, "spawn")
	)
	_worker_inputs: list[Connection] = field(init=False, factory=list)
	_worker_outputs: list[Connection] = field(init=False, factory=list)
	_worker_locks: list[Lock] = field(init=False, factory=list)
	_worker_log_queues: list[SimpleQueue[LogRecord]] = field(
		init=False, factory=list
	)
	_worker_log_threads: list[Thread] = field(init=False, factory=list)

	def _setup_workers(self, engine):
		super()._setup_workers(engine)
		wp = self._worker_processes
		wi = self._worker_inputs
		wo = self._worker_outputs
		wlk = self._worker_locks
		wl = self._worker_log_queues
		wlt = self._worker_log_threads
		assert len(wp) == len(wi) == len(wo) == len(wlk) == len(wlt)
		ctx = self._mp_ctx
		for i in range(engine.workers - len(wp)):
			inpipe_there, inpipe_here = ctx.Pipe(duplex=False)
			outpipe_here, outpipe_there = ctx.Pipe(duplex=False)
			logq = ctx.SimpleQueue()
			logthread = Thread(
				target=self._sync_log_forever,
				args=(self.logger, logq),
				daemon=True,
			)
			proc = ctx.Process(
				target=worker_subprocess,
				args=(
					i,
					engine._prefix,
					self._worker_last_branches,
					self._worker_last_eternal,
					engine.random_seed,
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

	def _send_worker_input_bytes(self, i: int, input: bytes) -> None:
		self._worker_inputs[i].send_bytes(input)

	def _get_worker_output_bytes(self, i: int) -> bytes:
		return self._worker_outputs[i].recv_bytes()

	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		super().shutdown(wait, cancel_futures=cancel_futures)
		todo = (
			list(range(len(self._worker_locks))),
			self._worker_locks,
			self._worker_inputs,
			self._worker_outputs,
			self._worker_processes,
			self._worker_log_queues,
			self._worker_log_threads,
		)
		while any(todo):
			(i, lock, pipein, pipeout, proc, logq, logt) = (
				some.pop() for some in todo
			)
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


@define
class LisienInterpreterExecutor(LisienExecutor):
	_worker_interpreters: list["concurrent.interpreters.Interpreter"] = field(
		init=False, factory=list
	)
	_worker_inputs: list[SimpleQueue[bytes]] = field(init=False, factory=list)
	_worker_outputs: list[SimpleQueue[bytes]] = field(init=False, factory=list)
	_worker_threads: list[Thread] = field(init=False, factory=list)
	_worker_locks: list[Lock] = field(init=False, factory=list)
	_worker_log_threads: list[Thread] = field(init=False, factory=list)
	_worker_log_queues: list[SimpleQueue[LogRecord]] = field(
		init=False, factory=list
	)

	def _setup_workers(self, engine) -> None:
		super()._setup_workers(engine)
		from concurrent.interpreters import (
			Interpreter,
			Queue,
			create,
			create_queue,
		)

		wint = self._worker_interpreters
		wi = self._worker_inputs
		wo = self._worker_outputs
		wt = self._worker_threads
		wlk = self._worker_locks
		wlq = self._worker_log_queues
		wlt = self._worker_log_threads
		i = None
		for i in range(self.workers - len(wint)):
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
				target=self._sync_log_forever,
				args=(self.logger, logq),
				daemon=True,
			)
			logthread.start()
			wlt.append(logthread)
			input.put(b"shutdown")
			terp_args = (
				worker_subinterpreter,
				i,
				engine._prefix,
				self._worker_last_branches,
				self._worker_last_eternal,
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

	def _send_worker_input_bytes(self, i: int, input: bytes) -> None:
		self._worker_inputs[i].put(input)

	def _get_worker_output_bytes(self, i: int) -> bytes:
		return self._worker_outputs[i].get()

	def shutdown(
		self, wait: bool = True, *, cancel_futures: bool = False
	) -> None:
		super().shutdown(wait, cancel_futures=cancel_futures)
		todo = (
			self._worker_locks,
			self._worker_inputs,
			self._worker_outputs,
			self._worker_threads,
			self._worker_interpreters,
			self._worker_log_queues,
			self._worker_log_threads,
		)
		while any(todo):
			(lock, inq, outq, thread, terp, logq, logt) = (
				some.pop() for some in todo
			)
			with lock:
				inq.put(b"shutdown")
				if logt.is_alive():
					logq.put(b"shutdown")
					logt.join(timeout=SUBPROCESS_TIMEOUT)
				thread.join(timeout=SUBPROCESS_TIMEOUT)
				if terp.is_running():
					terp.close()


@define
class LisienExecutorProxy(_BaseLisienExecutor):
	"""A :class:`LisienExecutor` that can itself be shared between processes

	Computation is performed in the host process, but all processes may
	submit work here.

	"""

	executor_class: ClassVar[type[LisienExecutor]]
	engine: AbstractEngine = field()

	@engine.validator
	def _validate_engine(self, _, engine):
		self._real = self.executor_class(engine)

	_real: executor_class = field(init=False)
	_pipe_here: Connection = field(init=False)
	_pipe_there: Connection = field(init=False)

	def _make_listen_thread(self):
		return Thread(target=self._listen_here, daemon=True)

	_listen_thread: Thread = field(
		init=False, default=Factory(_make_listen_thread, takes_self=True)
	)

	@property
	def lock(self) -> Lock:
		return self._real.lock

	@property
	def _worker_inputs(self) -> list[Connection] | list[SimpleQueue[bytes]]:
		return self._real._worker_inputs

	@property
	def _worker_locks(self) -> list[Lock]:
		return self._real._worker_locks

	@property
	def _worker_last_eternal(self) -> dict[EternalKey, Value]:
		return self._real._worker_last_eternal

	@_worker_last_eternal.setter
	def _worker_last_eternal(self, v):
		if not isinstance(v, dict):
			raise TypeError("Invalid eternal dict", v)
		self._real._worker_last_eternal = v

	def _setup_workers(self, engine):
		self._real = self.executor_class(engine)
		self._pipe_here, self._pipe_there = Pipe()
		try:
			self._listen_thread.start()
		except RuntimeError:
			self._listen_thread = self._make_listen_thread()
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
		if hasattr(self, "_pipe_there"):
			self._pipe_there.send(
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

	def submit(self, fn, /, *args, **kwargs):
		return self._real.submit(fn, *args, **kwargs)

	def restart(self, keyframe_cb: Callable[[], bytes] | None = None):
		if not hasattr(self, "_real"):
			self._setup_workers(self.engine)
		self._real.restart(keyframe_cb)

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


@define
class LisienThreadExecutorProxy(LisienExecutorProxy):
	executor_class = LisienThreadExecutor


@define
class LisienProcessExecutorProxy(LisienExecutorProxy):
	executor_class = LisienProcessExecutor


@define
class LisienInterpreterExecutorProxy(LisienExecutorProxy):
	executor_class = LisienInterpreterExecutor
