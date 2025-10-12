from __future__ import annotations

from enum import Enum
import logging
import os
import pickle
import sys
from threading import Thread

import msgpack
import tblib

from .engine import EngineProxy
from .process import engine_subprocess, engine_subthread


class Sub(Enum):
	process = "process"
	interpreter = "interpreter"
	thread = "thread"


class EngineProcessManager:
	"""Container for a Lisien proxy and a logger for it

	Make sure the :class:`EngineProcessManager` instance lasts as long as the
	:class:`lisien.proxy.EngineProxy` returned from its :method:`start`
	method. Call the :method:`EngineProcessManager.shutdown` method
	when you're done with the :class:`lisien.proxy.EngineProxy`. That way,
	we can join the thread that listens to the subprocess's logs.

	:param sub_mode: What form the subprocess should take, ``Sub.process`` by
	default. ``Sub.thread`` is more widely available, but doesn't allow
	true parallelism. On Python 3.14 and later, ``Sub.interpreter`` is available.

	"""

	loglevel = logging.DEBUG

	def __init__(
		self,
		*args,
		sub_mode: Sub = Sub.process,
		**kwargs,
	):
		self.sub_mode = Sub(sub_mode)
		self._args = args
		self._kwargs = kwargs

	def start(self, *args, **kwargs):
		"""Start lisien in a subprocess, and return a proxy to it"""
		if hasattr(self, "engine_proxy"):
			raise RuntimeError("Already started")
		try:
			import android

			android = True
		except ImportError:
			from multiprocessing import Pipe, SimpleQueue

			android = False
			(self._handle_in_pipe, self._proxy_out_pipe) = Pipe(duplex=False)
			(self._proxy_in_pipe, self._handle_out_pipe) = Pipe(duplex=False)
			self._logq = SimpleQueue()

		handlers = []
		logl = {
			"debug": logging.DEBUG,
			"info": logging.INFO,
			"warning": logging.WARNING,
			"error": logging.ERROR,
			"critical": logging.CRITICAL,
		}
		loglevel = self.loglevel
		if "loglevel" in kwargs:
			if kwargs["loglevel"] in logl:
				loglevel = logl[kwargs["loglevel"]]
			else:
				loglevel = kwargs["loglevel"]
			del kwargs["loglevel"]
		if "logger" in kwargs:
			self.logger = kwargs["logger"]
		else:
			self.logger = logging.getLogger(__name__)
			stdout = logging.StreamHandler(sys.stdout)
			stdout.set_name("stdout")
			handlers.append(stdout)
			handlers[0].setLevel(loglevel)
		if "logfile" in kwargs:
			try:
				fh = logging.FileHandler(kwargs["logfile"])
				handlers.append(fh)
				handlers[-1].setLevel(loglevel)
			except OSError:
				pass
			del kwargs["logfile"]
		replay_file = kwargs.pop("replay_file", "") or None
		install_modules = (
			kwargs.pop("install_modules")
			if "install_modules" in kwargs
			else []
		)
		enforce_end_of_time = kwargs.get("enforce_end_of_time", False)
		formatter = logging.Formatter(
			fmt="[{levelname}] lisien.proxy({process}) {message}", style="{"
		)
		for handler in handlers:
			handler.setFormatter(formatter)
			self.logger.addHandler(handler)
		match self.sub_mode:
			case Sub.process:
				self._start_subprocess()
			case Sub.thread:
				self._start_subthread()
			case Sub.interpreter:
				self._start_subinterpreter()
		self._make_proxy(args, kwargs)
		self.engine_proxy._init_pull_from_core()
		return self.engine_proxy

	def _sync_log_forever(self):
		while True:
			logrec = self._logq.get()
			self.logger.handle(self._undictify_logrec_traceback(logrec))

	def _undictify_logrec_traceback(
		self, logrec: logging.LogRecord
	) -> logging.LogRecord:
		if logrec.exc_info:
			if isinstance(logrec.exc_info, Exception):
				logrec.exc_info.__traceback__ = tblib.Traceback.from_dict(
					logrec.exc_info.__traceback__
				).as_traceback()
			elif (
				isinstance(logrec.exc_info, tuple)
				and len(logrec.exc_info) == 3
				and logrec.exc_info[2]
			):
				logrec.exc_info = (
					logrec.exc_info[0],
					logrec.exc_info[1],
					tblib.Traceback.from_dict(
						logrec.exc_info[2]
					).as_traceback(),
				)
		return logrec

	def _handle_log_record(self, _, logrec_packed: bytes):
		self.logger.handle(
			self._undictify_logrec_traceback(pickle.loads(logrec_packed))
		)

	def _initialize_proxy_db(self, **kwargs):
		branches_d = {"trunk": (None, 0, 0, 0, 0)}
		eternal_d = {
			"branch": "trunk",
			"turn": 0,
			"tick": 0,
			"_lisien_schema_version": 0,
		}

		if "connect_string" in kwargs:
			from sqlalchemy import NullPool, create_engine, select
			from sqlalchemy.exc import OperationalError

			from ..alchemy import meta

			eng = create_engine(
				kwargs["connect_string"],
				poolclass=NullPool,
				**kwargs.get("connect_args", {}),
			)
			conn = eng.connect()
			branches_t = meta.tables["branches"]
			branches_sel = select(
				branches_t.c.branch,
				branches_t.c.parent,
				branches_t.c.parent_turn,
				branches_t.c.parent_tick,
				branches_t.c.end_turn,
				branches_t.c.end_tick,
			)
			try:
				for (
					branch,
					parent,
					parent_turn,
					parent_tick,
					end_turn,
					end_tick,
				) in conn.execute(branches_sel):
					branches_d[branch] = (
						parent,
						parent_turn,
						parent_tick,
						end_turn,
						end_tick,
					)
			except OperationalError:
				pass
			global_t = meta.tables["global"]
			eternal_sel = select(global_t.c.key, global_t.c.value)
			try:
				for key, value in conn.execute(eternal_sel):
					eternal_d[key] = value
			except OperationalError:
				pass
		else:
			from parquetdb import ParquetDB

			for d in (
				ParquetDB("branches")
				.read(
					columns=[
						"parent",
						"parent_turn",
						"parent_tick",
						"end_turn",
						"end_tick",
					]
				)
				.to_pylist()
			):
				branches_d[d["branch"]] = (
					d["parent"],
					d["parent_turn"],
					d["parent_tick"],
					d["end_turn"],
					d["end_tick"],
				)
			for d in (
				ParquetDB("global").read(columns=["key", "value"]).to_pylist()
			):
				eternal_d[d["key"]] = d["value"]
			return branches_d, eternal_d

	def log(self, level: str | int, msg: str):
		if isinstance(level, str):
			level = {
				"debug": 10,
				"info": 20,
				"warning": 30,
				"error": 40,
				"critical": 50,
			}[level.lower()]
		self.logger.log(level, msg)

	def shutdown(self):
		"""Close the engine in the subprocess, then join the subprocess"""
		if hasattr(self, "engine_proxy"):
			self.engine_proxy.close()
			if hasattr(self, "_p"):
				self.engine_proxy.send_bytes(b"shutdown")
				self._p.join(timeout=1)
			del self.engine_proxy
		if hasattr(self, "_server"):
			self._server.shutdown()

	def _config_logger(self, kwargs):
		handlers = []
		logl = {
			"debug": logging.DEBUG,
			"info": logging.INFO,
			"warning": logging.WARNING,
			"error": logging.ERROR,
			"critical": logging.CRITICAL,
		}
		loglevel = self.loglevel
		if "loglevel" in kwargs:
			if kwargs["loglevel"] in logl:
				loglevel = logl[kwargs["loglevel"]]
			else:
				loglevel = kwargs["loglevel"]
			del kwargs["loglevel"]
		if "logger" in kwargs:
			self.logger = kwargs["logger"]
		else:
			self.logger = logging.getLogger(__name__)
			stdout = logging.StreamHandler(sys.stdout)
			stdout.set_name("stdout")
			handlers.append(stdout)
			handlers[0].setLevel(loglevel)
		if "logfile" in kwargs:
			try:
				fh = logging.FileHandler(kwargs.pop("logfile"))
				handlers.append(fh)
				handlers[-1].setLevel(loglevel)
			except OSError:
				pass
		formatter = logging.Formatter(
			fmt="[{levelname}] lisien.proxy({process}) {message}", style="{"
		)
		for handler in handlers:
			handler.setFormatter(formatter)
			self.logger.addHandler(handler)

	def _start_subprocess(self, *args, **kwargs):
		from multiprocessing import Pipe, Process, SimpleQueue

		(self._handle_in_pipe, self._proxy_out_pipe) = Pipe(duplex=False)
		(self._proxy_in_pipe, self._handle_out_pipe) = Pipe(duplex=False)
		self._logq = SimpleQueue()

		self._p = Process(
			name="Lisien Life Simulator Engine (core)",
			target=engine_subprocess,
			args=(
				args or self._args,
				self._kwargs | kwargs,
				self._handle_in_pipe,
				self._handle_out_pipe,
			),
			kwargs={
				"log_queue": self._logq,
			},
		)
		self._p.start()

		self._log_thread = Thread(target=self._sync_log_forever, daemon=True)
		self._log_thread.start()

	def _start_subthread(self, *args, **kwargs):
		from queue import SimpleQueue

		self._input_queue = SimpleQueue()
		self._output_queue = SimpleQueue()

		self._t = Thread(
			target=engine_subthread,
			args=(
				args or self._args,
				self._kwargs | kwargs,
				self._input_queue,
				self._output_queue,
			),
		)
		self._t.start()

	def _start_subinterpreter(self, *args, **kwargs):
		from concurrent.interpreters import create, create_queue
		from queue import Queue

		self._input_queue: Queue = create_queue()
		self._output_queue: Queue = create_queue()

		self._terp = create()
		self._t = self._terp.call_in_thread(
			engine_subthread,
			args=(
				args or self._args,
				self._kwargs | kwargs,
				self._input_queue,
				self._output_queue,
			),
		)

	def _make_proxy(
		self, *args, install_modules=(), enforce_end_of_time=False, **kwargs
	):
		self._config_logger(kwargs)
		branches_d, eternal_d = self._initialize_proxy_db(**kwargs)
		if hasattr(self, "_handle_in_pipe") and hasattr(
			self, "_handle_out_pipe"
		):
			self.engine_proxy = EngineProxy(
				self._proxy_in_pipe.recv_bytes,
				self._proxy_out_pipe.send_bytes,
				self.logger,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
				branches=branches_d,
				eternal=eternal_d,
			)
		else:
			self.engine_proxy = EngineProxy(
				self._input_queue.get,
				self._output_queue.put,
				self.logger,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
				branches=branches_d,
				eternal=eternal_d,
			)

		return self.engine_proxy

	def load_archive(
		self,
		archive_path: str | os.PathLike,
		prefix: str | os.PathLike,
		**kwargs,
	) -> EngineProxy:
		"""Load a game from a .lisien archive, start Lisien on it, and return its proxy"""
		n = len(".lisien")
		if archive_path[-n:] != ".lisien":
			raise RuntimeError("Not a .lisien archive")
		self._start_subprocess()
		self._config_logger(kwargs)
		if hasattr(self, "_proxy_in_pipe"):
			self._proxy_in_pipe.send(
				b"from_archive"
				+ msgpack.packb(
					{"archive_path": archive_path, "prefix": prefix, **kwargs}
				)
			)
		else:
			self._input_queue.put(
				(
					"from_archive",
					{"archive_path": archive_path, "prefix": prefix, **kwargs},
				)
			)
		self._make_proxy()
		self.engine_proxy._init_pull_from_core()
		return self.engine_proxy

	def __enter__(self):
		return self.start()

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.shutdown()
