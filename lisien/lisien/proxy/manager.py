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

import ast
import json
import logging
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from queue import Empty
from threading import Lock, Thread
from zipfile import ZipFile

import tblib
from attrs import define, field

from ..enum import Sub
from ..facade import EngineFacade
from ..types import Branch, EternalKey, Key, Tick, Turn, Value
from ..util import unpack_expected
from .engine import EngineProxy
from .routine import engine_subprocess, engine_subthread

TIMEOUT = 60


@define(slots=False)
class EngineProxyManager:
	"""Container for a Lisien proxy and a logger for it

	Make sure the :class:`EngineProxyManager` instance lasts as long as the
	:class:`lisien.proxy.EngineProxy` returned from its :method:`start`
	method. Call the :method:`EnginePrxyManager.shutdown` method
	when you're done with the :class:`lisien.proxy.EngineProxy`. That way,
	we can join the thread that listens to the subprocess's logs.

	"""

	sub_mode: Sub = field(converter=Sub, default=Sub.thread)
	"""What form the subprocess should take. ``Sub.thread``
	is the most widely available, and is therefore the default, but doesn't
	allow true parallelism unless you're running a GIL-less build of Python.
	``Sub.process`` does allow true parallelism, but isn't available on Android.
	``Sub.interpreter`` is, and allows true parallelism as well, but is only
	available on Python 3.14 or later."""

	@sub_mode.validator
	def _validate_sub_mode(self, _, sub_mode):
		if sub_mode is None or sub_mode is Sub.serial:
			raise NotImplementedError(
				"Can't presently run proxies without some subroutine"
			)

	loglevel: int = logging.DEBUG
	"""What level to log at"""
	android: bool = field(default=False)
	"""Are we running on Android?"""
	reuse: bool = field(default=False)
	"""Whether to keep the subprocess running after closing the engine proxy.
	
	Or the subthread, or the subinterpreter. Whatever you're using.
	
	Starting a new engine in the same subprocess is often faster.
	
	"""
	_top_uid: int = field(init=False, default=0)
	_proxman_send_lock: Lock = field(init=False, factory=Lock)
	_proxman_recv_lock: Lock = field(init=False, factory=Lock)
	_round_trip_lock: Lock = field(init=False, factory=Lock)
	logger: logging.Logger = field(
		init=False, factory=partial(logging.getLogger, __name__)
	)

	def start(self, prefix: str | None = None, **kwargs):
		self._config_logger(kwargs)
		if "sub_mode" in kwargs:
			kwargs["sub_mode"] = Sub(kwargs["sub_mode"]).value

		if self.android:
			self._start_subthread(prefix, **kwargs)
		else:
			match self.sub_mode:
				case Sub.process:
					self._start_subprocess(prefix, **kwargs)
				case Sub.thread:
					self._start_subthread(prefix, **kwargs)
				case Sub.interpreter:
					self._start_subinterpreter(prefix, **kwargs)
				case Sub.none:
					raise NotImplementedError("Just use Engine")
		if prefix and "prefix" in kwargs:
			raise TypeError(
				"Got multiple arguments for prefix", prefix, kwargs["prefix"]
			)
		elif "prefix" in kwargs:
			prefix = kwargs.pop("prefix")
		if prefix is not None:
			prefix = Path(prefix)
		branch, turn, tick, branches_d, eternal_d = self._go_time()
		self._make_proxy(
			branch, turn, tick, branches_d, eternal_d, prefix, **kwargs
		)
		self.engine_proxy._init_pull_from_core()
		return self.engine_proxy

	def _sync_log_forever(self):
		while (logrec := self._logq.get()) != b"shutdown":
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
			if not self.engine_proxy.closed:
				self.engine_proxy.close()
			unpack = self.engine_proxy.unpack
		else:
			eng = EngineFacade(None)
			unpack = eng.unpack
		if hasattr(self, "_p"):
			if self._p.is_alive():
				with self._round_trip_lock:
					with self._proxman_send_lock:
						self._proxman_send_pipe.send_bytes(b"shutdown")
					with self._proxman_recv_lock:
						if not self._proxman_recv_pipe.poll(TIMEOUT):
							raise TimeoutError(
								"No response to shutdown signal"
							)
						got = self._proxman_recv_pipe.recv_bytes()
						unpack_expected(unpack, got, b"shutdown")
				self._p.join(timeout=TIMEOUT)
				if self._p.is_alive():
					self._p.kill()
					self._p.join(timeout=TIMEOUT)
					if self._p.is_alive():
						self._p.terminate()
				self._p.close()
			del self._p
		if hasattr(self, "_t"):
			if self._t.is_alive():
				with self._round_trip_lock:
					with self._proxman_send_lock:
						self._proxman_put_queue.put(b"shutdown")
					with self._proxman_recv_lock:
						try:
							got: bytes = self._proxman_get_queue.get(
								timeout=TIMEOUT
							)
						except Empty:
							raise TimeoutError(
								"Didn't get a timely response from the core thread"
							)
					unpack_expected(unpack, got, b"shutdown")
				self._t.join(timeout=TIMEOUT)
				if self._t.is_alive():
					raise TimeoutError("Couldn't join thread")
			del self._t

		if hasattr(self, "_logq"):
			self._logq.put(b"shutdown")
		if hasattr(self, "_log_thread"):
			self._log_thread.join()
			del self._log_thread
		if hasattr(self, "_logq"):
			del self._logq
		if hasattr(self, "_terp"):
			if self._terp.is_running():
				self._terp.close()
			del self._terp
		if hasattr(self, "_client"):
			while not self._proxman_get_queue.empty():
				time.sleep(0.01)
		if hasattr(self, "_server"):
			self._server.shutdown()
		if hasattr(self, "engine_proxy"):
			del self.engine_proxy
		if hasattr(self, "logger"):
			self.logger.debug("EngineProxyManager: shutdown")

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

	def _start_subprocess(self, prefix: Path | None = None, **kwargs):
		if hasattr(self, "_p"):
			if not self.reuse:
				raise RuntimeError("Already started")
			if not self._p.is_alive():
				raise RuntimeError("Tried to reuse a dead process")
			self.logger.info("EngineProxyManager: already have a subprocess")
			if hasattr(self, "engine_proxy"):
				if not self.engine_proxy.closed:
					raise RuntimeError("Previous engine proxy was not closed")
				pack = self.engine_proxy.pack
				unpack = self.engine_proxy.unpack
			else:
				eng = EngineFacade(None)
				pack = eng.pack
				unpack = eng.unpack
			with self._round_trip_lock:
				with self._proxman_send_lock:
					self._proxman_send_pipe.send_bytes(
						pack(
							{"command": "restart", "prefix": prefix, **kwargs}
						)
					)
				with self._proxman_recv_lock:
					if not self._proxman_recv_pipe.poll(TIMEOUT):
						raise TimeoutError(
							"Subprocess didn't respond to restart command"
						)
					if not (
						got := self._proxman_recv_pipe.recv_bytes()
					).endswith(b"\xa9restarted"):
						unpack_expected(unpack, got, b"\xa9restarted")
			return
		from multiprocessing import Pipe, Process, SimpleQueue

		(self._handle_recv_pipe, self._proxman_send_pipe) = Pipe(duplex=False)
		(self._proxman_recv_pipe, self._handle_send_pipe) = Pipe(duplex=False)
		self._logq = SimpleQueue()

		self._p = Process(
			name="Lisien Life Simulator Engine (core)",
			target=engine_subprocess,
			args=(
				(prefix,),
				kwargs,
				self._handle_recv_pipe,
				self._handle_send_pipe,
				self.reuse,
				self._logq,
			),
		)
		self._p.start()

		self._log_thread = Thread(target=self._sync_log_forever)
		self._log_thread.start()

	def _start_subthread(self, prefix: Path | None = None, **kwargs):
		if hasattr(self, "_t"):
			if not self.reuse:
				raise RuntimeError("Already started")
			self.logger.info(
				"EngineProxyManager: already have a subthread, will reuse"
			)
			if hasattr(self, "engine_proxy"):
				if not self.engine_proxy.closed:
					raise RuntimeError("Previous engine proxy was not closed")
				pack = self.engine_proxy.pack
				unpack = self.engine_proxy.unpack
			else:
				eng = EngineFacade(None)
				pack = eng.pack
				unpack = eng.unpack
			restart_bytes = pack(
				{"command": "restart", "prefix": prefix, **kwargs}
			)
			with self._round_trip_lock:
				self._proxman_put_queue.put(restart_bytes)
				if not (got := self._proxman_get_queue.get()).endswith(
					b"\xa9restarted"
				):
					try:
						gotten = unpack(got)
						if isinstance(gotten, Exception):
							raise gotten
						if not isinstance(gotten, tuple) or len(gotten) == 0:
							raise TypeError(
								"Strange output from subthread", gotten
							)
						if isinstance(gotten[-1], Exception):
							raise gotten[-1]
						else:
							raise RuntimeError(
								"Failed to restart subthread", gotten
							)
					except ValueError:
						raise RuntimeError("Failed to restart subthread", got)
			return
		self.logger.debug("EngineProxyManager: starting subthread!")
		from queue import SimpleQueue

		self._proxman_put_queue = SimpleQueue()
		self._proxman_get_queue = SimpleQueue()

		self._t = Thread(
			target=engine_subthread,
			args=(
				(prefix,),
				kwargs,
				self._proxman_put_queue,
				self._proxman_get_queue,
				self.reuse,
				None,
			),
		)
		self._t.start()

	def _start_subinterpreter(self, prefix: Path | None = None, **kwargs):
		if hasattr(self, "_terp"):
			if not self.reuse:
				raise RuntimeError("Already started")
			self.logger.info(
				"EngineProxyManager: already have a subinterpreter"
			)
			if hasattr(self, "engine_proxy"):
				if not self.engine_proxy.closed:
					raise RuntimeError("Previous engine proxy was not closed")
				pack = self.engine_proxy.pack
				unpack = self.engine_proxy.unpack
			else:
				eng = EngineFacade(None)
				pack = eng.pack
				unpack = eng.unpack
			self._proxman_put_queue.put(
				pack({"command": "restart", "prefix": prefix, **kwargs})
			)
			if not (got := self._proxman_get_queue.get()).endswith(
				b"\xa9restarted"
			):
				gotten = unpack(got)
				if not isinstance(gotten, tuple) or len(gotten) == 0:
					raise TypeError(
						"Strange output from subinterpreter", gotten
					)
				if isinstance(gotten[-1], Exception):
					raise gotten[-1]
				raise RuntimeError("Failed to restart subinterpreter", got)
			return
		from concurrent.interpreters import create, create_queue
		from queue import Queue

		self._proxman_put_queue: Queue = create_queue()
		self._proxman_get_queue: Queue = create_queue()
		self._logq: Queue = create_queue()

		self._terp = create()
		self._t = self._terp.call_in_thread(
			engine_subthread,
			(prefix,),
			kwargs,
			self._proxman_put_queue,
			self._proxman_get_queue,
			self._logq,
		)
		self._log_thread = Thread(target=self._sync_log_forever)
		self._log_thread.start()

	def _preload(
		self, prefix: Path | None, **kwargs
	) -> tuple[
		dict[Branch, tuple[Branch, Turn, Tick, Turn, Tick]],
		dict[EternalKey, Value],
	]:
		branches_d: dict[Branch, tuple[Branch, Turn, Tick, Turn, Tick]] = {
			"trunk": (None, 0, 0, 0, 0)
		}
		eternal_d: dict[EternalKey, Value] = {
			EternalKey(Key("branch")): Value("trunk"),
			EternalKey(Key("turn")): Value(0),
			EternalKey(Key("tick")): Value(0),
			EternalKey(Key("_lisien_schema_version")): Value(0),
		}
		if "database" in kwargs:
			if isinstance(kwargs["database"], partial):
				which_db = kwargs["database"].func.db_type
			else:
				which_db = kwargs["database"].db_type
		elif "connect_string" in kwargs:
			which_db = "sql"
		elif prefix is None:
			which_db = "python"
		else:
			try:
				import parquetdb

				which_db = "parquetdb"
			except ImportError:
				kwargs["connect_string"] = f"sqlite:///{prefix}/world.sqlite3"
				which_db = "sql"

		self.logger.debug(f"initializing a proxy with database: {which_db}")

		if which_db == "sql":
			from sqlalchemy import NullPool, create_engine, select
			from sqlalchemy.exc import OperationalError

			from ..sql import meta

			connect_args = {}
			if "connect_string" in kwargs:
				connect_string = kwargs["connect_string"]
				if "connect_args" in kwargs:
					connect_args = kwargs["connect_args"]
			elif "database" in kwargs and isinstance(
				kwargs["database"], partial
			):
				connect_string = kwargs["database"].keywords["connect_string"]
				if "connect_args" in kwargs["database"].keywords:
					connect_args = kwargs["database"].keywords["connect_args"]
			elif prefix is not None:
				connect_string = f"sqlite:///{prefix}/world.sqlite3"
			else:
				raise ValueError("Don't have a connect_string for SQL")
			eng = create_engine(
				connect_string,
				poolclass=NullPool,
				**connect_args,
			)
			branches_t = meta.tables["branches"]
			branches_sel = select(
				branches_t.c.branch,
				branches_t.c.parent,
				branches_t.c.parent_turn,
				branches_t.c.parent_tick,
				branches_t.c.end_turn,
				branches_t.c.end_tick,
			)
			global_t = meta.tables["global"]
			eternal_sel = select(global_t.c.key, global_t.c.value)
			with eng.connect() as conn:
				try:
					branches_data = conn.execute(branches_sel).fetchall()
				except OperationalError:
					branches_data = []
				for (
					branch,
					parent,
					parent_turn,
					parent_tick,
					end_turn,
					end_tick,
				) in branches_data:
					branches_d[branch] = (
						parent,
						parent_turn,
						parent_tick,
						end_turn,
						end_tick,
					)
				try:
					eternal_data = conn.execute(eternal_sel).fetchall()
				except OperationalError:
					eternal_data = []
				for key, value in eternal_data:
					eternal_d[key] = value
		elif which_db == "python":
			self.logger.warning(
				"Running without a database. Lisien will be empty at start."
			)
		elif which_db == "parquetdb":
			from parquetdb import ParquetDB

			if "database" in kwargs and isinstance(
				kwargs["database"], partial
			):
				pqdb_prefix = kwargs["database"].keywords["path"]
			else:
				pqdb_prefix = prefix.joinpath("world")

			for d in (
				ParquetDB(pqdb_prefix.joinpath("branches"))
				.read(
					columns=[
						"branch",
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
				ParquetDB(f"{pqdb_prefix}/global")
				.read(columns=["key", "value"])
				.to_pylist()
			):
				eternal_d[d["key"]] = d["value"]
		else:
			raise ValueError("Couldn't determine database type", which_db)
		return branches_d, eternal_d

	def _go_time(
		self,
	) -> tuple[
		Branch,
		Turn,
		Tick,
		dict[Branch, tuple[Branch, Turn, Tick, Turn, Tick]],
		dict[EternalKey, Value],
	]:
		if hasattr(self, "engine_proxy"):
			if not self.engine_proxy.closed:
				raise RuntimeError(
					"Tried to make a second proxy in EngineProxyManager"
				)
			unpack = self.engine_proxy.unpack
		else:
			unpack = EngineFacade(None).unpack
		with self._round_trip_lock:
			if hasattr(self, "_proxman_put_queue"):
				with self._proxman_send_lock:
					self._proxman_put_queue.put(b"go")
				with self._proxman_recv_lock:
					try:
						got = self._proxman_get_queue.get(timeout=TIMEOUT)
					except Empty:
						raise TimeoutError("Core didn't respond")
			else:
				with self._proxman_send_lock:
					self._proxman_send_pipe.send_bytes(b"go")
				if not self._proxman_recv_pipe.poll(TIMEOUT):
					raise TimeoutError("Subprocess isn't ready")
				with self._proxman_recv_lock:
					got = self._proxman_recv_pipe.recv_bytes()
			gotten = unpack(got)
			if isinstance(gotten, BaseException):
				raise gotten
			cmd, branch, turn, tick, (branches_d, eternal_d) = gotten
			if not isinstance(cmd, str) or not cmd.startswith(
				"handle initialized"
			):
				raise RuntimeError("Failed to initialize EngineHandle", cmd)
		return Branch(branch), Turn(turn), Tick(tick), branches_d, eternal_d

	def _make_proxy(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		branches_d: dict,
		eternal_d: dict,
		prefix: Path | None,
		install_modules=(),
		enforce_end_of_time=False,
		game_source_code: dict[str, str] | None = None,
		game_strings: dict[str, str] | None = None,
		**kwargs,
	):
		if game_source_code is None:
			game_source_code = {}
			if prefix is not None:
				prefix = Path(prefix)
				for store in (
					"function",
					"method",
					"trigger",
					"prereq",
					"action",
				):
					pyfile = prefix.joinpath(store + ".py")
					if pyfile.exists() and pyfile.stat().st_size:
						code = game_source_code[store] = {}
						with open(pyfile, "rt") as inf:
							parsed = ast.parse(inf.read(), pyfile)
						funk: ast.FunctionDef
						for funk in parsed.body:
							code[funk.name] = ast.unparse(funk)
		if game_strings is None:
			if prefix and prefix.joinpath("strings").is_dir():
				lang = eternal_d.get(EternalKey(Key("language")), "eng")
				jsonpath = prefix.joinpath("strings", str(lang) + ".json")
				if jsonpath.is_file():
					with open(jsonpath) as inf:
						game_strings = json.load(inf)

		if hasattr(self, "_proxman_recv_pipe") and hasattr(
			self, "_proxman_send_pipe"
		):
			self.engine_proxy = EngineProxy(
				branch,
				turn,
				tick,
				self._proxman_recv_pipe.recv_bytes,
				self._proxman_send_pipe.send_bytes,
				self.logger,
				self._proxman_recv_lock,
				self._proxman_send_lock,
				self._round_trip_lock,
				prefix,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
				branches_d=branches_d,
				eternal=eternal_d,
				strings=game_strings,
				**game_source_code,
			)
		else:
			self.engine_proxy = EngineProxy(
				branch,
				turn,
				tick,
				self._proxman_get_queue.get,
				self._proxman_put_queue.put,
				self.logger,
				self._proxman_recv_lock,
				self._proxman_send_lock,
				self._round_trip_lock,
				prefix,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
				branches_d=branches_d,
				eternal=eternal_d,
				strings=game_strings,
				**game_source_code,
			)

		return self.engine_proxy

	def load_archive(
		self,
		archive_path: str | os.PathLike,
		prefix: str | os.PathLike,
		**kwargs,
	) -> EngineProxy:
		"""Load a game from a .lisien archive, start Lisien on it, and return its proxy"""
		if isinstance(archive_path, Path):
			if not archive_path.name.endswith(".lisien"):
				raise RuntimeError("Not a .lisien archive")
		elif not archive_path.endswith(".lisien"):
			raise RuntimeError("Not a .lisien archive")
		archive_path = Path(archive_path)
		prefix = Path(prefix)
		game_code = {}
		with ZipFile(archive_path) as zf:
			namelist = zf.namelist()
			for pypre in ["function", "method", "trigger", "prereq", "action"]:
				pyfn = pypre + ".py"
				if pyfn in namelist:
					code = game_code[pypre] = {}
					with zf.open(pyfn, "r") as inf:
						parsed = ast.parse(inf.read().decode("utf-8"), pyfn)
					funk: ast.FunctionDef
					for funk in parsed.body:
						code[funk.name] = ast.unparse(funk)
		self._config_logger(kwargs)
		try:
			import android

			self._start_osc(prefix, **kwargs)
		except ModuleNotFoundError:
			pass
		match self.sub_mode:
			case Sub.interpreter:
				self._start_subinterpreter(prefix, **kwargs)
			case Sub.process:
				self._start_subprocess(prefix, **kwargs)
			case Sub.thread:
				self._start_subthread(prefix, **kwargs)
		worker_sub = kwargs.pop("sub_mode", None)
		if worker_sub is not None:
			worker_sub = Sub(worker_sub).value
		if hasattr(self, "engine_proxy"):
			pack = self.engine_proxy.pack
			unpack = self.engine_proxy.unpack
		else:
			fac = EngineFacade(None)
			pack = fac.pack
			unpack = fac.unpack
		payload = b"from_archive" + pack(
			{
				"archive_path": str(archive_path),
				"prefix": str(prefix),
				"sub_mode": worker_sub,
				**kwargs,
			}
		)
		with self._round_trip_lock:
			if hasattr(self, "_proxman_send_pipe"):
				with self._proxman_send_lock:
					self._proxman_send_pipe.send_bytes(payload)
				with self._proxman_recv_lock:
					if not self._proxman_recv_pipe.poll(TIMEOUT):
						raise TimeoutError("Didn't import archive in time")
					recvd = self._proxman_recv_pipe.recv_bytes()
			else:
				with self._proxman_send_lock:
					self._proxman_put_queue.put(payload)
				with self._proxman_recv_lock:
					try:
						recvd = self._proxman_get_queue.get(timeout=TIMEOUT)
					except Empty:
						raise TimeoutError("Didn't import archive in time")
		unpacked = unpack(recvd)
		if isinstance(unpacked, BaseException):
			raise unpacked
		cmd, branch, turn, tick, _ = unpacked
		if cmd != "handle initialized from archive":
			raise RuntimeError(
				f"Expected 'handle initialized from archive', got {cmd}"
			)
		branches_d, eternal_d = self._preload(prefix, **kwargs)
		self._make_proxy(
			branch,
			turn,
			tick,
			branches_d,
			eternal_d,
			prefix,
			game_source_code=game_code,
			**kwargs,
		)
		self.engine_proxy._init_pull_from_core()
		return self.engine_proxy

	def close(self):
		self.shutdown()
		if hasattr(self, "_client"):
			self._client.send_message("127.0.0.1/shutdown")
			self.logger.debug(
				"EngineProxyManager: joining input sender thread"
			)
			self._input_sender_thread.join()
			self.logger.debug("EngineProxyManager: joined input sender thread")
			self.logger.debug("EngineProxyManager: stopping core service")
			self._core_service.stop()
			self.logger.debug("EngineProxyManager: stopped core service")
		if hasattr(self, "logger"):
			self.logger.debug("EngineProxyManager: closed")

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()
