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
import zlib
from functools import partial
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import tblib

from ..facade import EngineFacade
from ..types import Branch, EternalKey, Key, Tick, Turn, Value, Sub
from .engine import EngineProxy
from .routine import engine_subprocess, engine_subthread


class EngineProxyManager:
	"""Container for a Lisien proxy and a logger for it

	Make sure the :class:`EngineProxyManager` instance lasts as long as the
	:class:`lisien.proxy.EngineProxy` returned from its :method:`start`
	method. Call the :method:`EnginePrxyManager.shutdown` method
	when you're done with the :class:`lisien.proxy.EngineProxy`. That way,
	we can join the thread that listens to the subprocess's logs.

	:param sub_mode: What form the subprocess should take. ``Sub.thread``
	is the most widely available, and is therefore the default, but doesn't
	allow true parallelism unless you're running a GIL-less build of Python.
	``Sub.process`` does allow true parallelism, but isn't available on Android.
	``Sub.interpreter`` is, and allows true parallelism as well, but is only
	available on Python 3.14 or later.

	"""

	loglevel = logging.DEBUG
	android = False
	_really_shutdown = True

	def __init__(self, sub_mode: Sub = Sub.thread):
		self.sub_mode = Sub(sub_mode)
		self._top_uid = 0

	def start(self, *args, **kwargs):
		self._config_logger(kwargs)

		if self.android:
			self._start_osc(*args, **kwargs)
		else:
			match self.sub_mode:
				case Sub.process:
					self._start_subprocess(*args, **kwargs)
				case Sub.thread:
					self._start_subthread(*args, **kwargs)
				case Sub.interpreter:
					self._start_subinterpreter(*args, **kwargs)
		if args and "prefix" in kwargs:
			raise TypeError(
				"Got multiple arguments for prefix", args[0], kwargs["prefix"]
			)
		elif args:
			prefix = args[0]
		elif "prefix" in kwargs:
			prefix = kwargs.pop("prefix")
		else:
			prefix = None
		if prefix is not None:
			prefix = Path(prefix)
		self._make_proxy(prefix, **kwargs)
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

	def _initialize_proxy_db(
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
				import ParquetDB

				which_db = "parquetdb"
			except ImportError:
				kwargs["connect_string"] = f"sqlite:///{prefix}/world.sqlite3"
				which_db = "sql"

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
		if hasattr(self, "engine_proxy") and not self.engine_proxy.closed:
			self.engine_proxy.close()
		if self._really_shutdown:
			if hasattr(self, "_logq"):
				self._logq.put(b"shutdown")
				del self._logq
			if hasattr(self, "_p"):
				if self._p.is_alive():
					self._handle_out_pipe.send_bytes(b"shutdown")
					if (
						got := self._proxy_in_pipe.recv_bytes()
					) != b"shutdown":
						raise RuntimeError(
							"Subprocess didn't respond to shutdown signal", got
						)
					self._p.join(timeout=10.0)
					if self._p.is_alive():
						self._p.kill()
						self._p.join(timeout=10.0)
						if self._p.is_alive():
							self._p.terminate()
					self._p.close()
				del self._p
			if hasattr(self, "_t"):
				if self._t.is_alive():
					self._input_queue.put(b"shutdown")
					if (
						got := self._output_queue.get(timeout=5.0)
					) != b"shutdown":
						raise RuntimeError(
							"Subthread didn't respond to shutdown signal", got
						)
					self._t.join(timeout=5.0)
					if self._t.is_alive():
						raise TimeoutError("Couldn't join thread")
				del self._t
			if hasattr(self, "_log_thread"):
				self._log_thread.join()
				del self._log_thread
			if hasattr(self, "_terp"):
				if self._terp.is_running():
					self._terp.close()
				del self._terp
		if hasattr(self, "_client"):
			while not self._output_queue.empty():
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

	def _start_subprocess(self, prefix: str | None = None, **kwargs):
		if hasattr(self, "_p"):
			if self._really_shutdown:
				raise RuntimeError("Already started")
			self.logger.info("EngineProxyManager: already have a subprocess")
			if hasattr(self, "engine_proxy"):
				pack = self.engine_proxy.pack
			else:
				pack = EngineFacade(None).pack
			self._proxy_out_pipe.send_bytes(
				pack({"command": "restart", "prefix": prefix, **kwargs})
			)
			if not (got := self._proxy_in_pipe.recv_bytes()).endswith(
				b"\xa9restarted"
			):
				raise RuntimeError("Failed to restart subprocess", got)
			return
		from multiprocessing import Pipe, Process, SimpleQueue

		(self._handle_in_pipe, self._proxy_out_pipe) = Pipe(duplex=False)
		(self._proxy_in_pipe, self._handle_out_pipe) = Pipe(duplex=False)
		self._logq = SimpleQueue()

		self._p = Process(
			name="Lisien Life Simulator Engine (core)",
			target=engine_subprocess,
			args=(
				(prefix,),
				kwargs,
				self._handle_in_pipe,
				self._handle_out_pipe,
				self._logq,
			),
		)
		self._p.start()

		self._log_thread = Thread(target=self._sync_log_forever)
		self._log_thread.start()

	def _start_osc(self, *args, **kwargs):
		if hasattr(self, "_core_service"):
			self.logger.info(
				"EngineProxyManager: reusing existing OSC core service at %s",
				self._core_service.server_address,
			)
			return
		import random
		from queue import SimpleQueue

		from android import autoclass
		from pythonosc.dispatcher import Dispatcher
		from pythonosc.osc_tcp_server import ThreadingOSCTCPServer
		from pythonosc.tcp_client import SimpleTCPClient

		low_port = 32000
		high_port = 65535
		core_port_queue = SimpleQueue()
		self._input_queue = SimpleQueue()
		self._output_queue = SimpleQueue()
		disp = Dispatcher()
		disp.map(
			"/core-report-port", lambda _, port: core_port_queue.put(port)
		)
		disp.map("/log", self._handle_log_record)
		self._input_received = []
		disp.map("/", self._receive_input)
		for _ in range(128):
			procman_port = random.randint(low_port, high_port)
			try:
				self._server = ThreadingOSCTCPServer(
					("127.0.0.1", procman_port), disp
				)
				self._server_thread = Thread(target=self._server.serve_forever)
				self._server_thread.start()
				self.logger.debug(
					"EngineProxyManager: started server at port %d",
					procman_port,
				)
				break
			except OSError:
				pass
		else:
			sys.exit("couldn't get port for process manager")

		mActivity = self._mActivity = autoclass(
			"org.kivy.android.PythonActivity"
		).mActivity
		core_service = self._core_service = autoclass(
			"org.tacmeta.elide.ServiceCore"
		)
		argument = repr(
			[
				low_port,
				high_port,
				procman_port,
				args or self._args,
				kwargs | self._kwargs,
			]
		)
		try:
			self.logger.debug("EngineProxyManager: starting core...")
			core_service.start(mActivity, argument)
		except Exception as ex:
			self.logger.critical(repr(ex))
			sys.exit(repr(ex))
		core_port = core_port_queue.get()
		self._client = SimpleTCPClient("127.0.0.1", core_port)
		self.logger.info(
			"EngineProxyManager: connected to lisien core over OSC at port %d",
			core_port,
		)

	def _send_output_forever(self, output_queue):
		from pythonosc.osc_message_builder import OscMessageBuilder

		assert hasattr(self, "engine_proxy"), (
			"EngineProxyManager tried to send input with no EngineProxy"
		)
		while True:
			cmd = output_queue.get()
			msg = zlib.compress(cmd)
			chunks = len(msg) // 1024
			if len(msg) % 1024:
				chunks += 1
			self.logger.debug(
				f"EngineProxyManager: about to send a command to core in {chunks} chunks"
			)
			for n in range(chunks):
				builder = OscMessageBuilder("/")
				builder.add_arg(self._top_uid, OscMessageBuilder.ARG_TYPE_INT)
				builder.add_arg(chunks, OscMessageBuilder.ARG_TYPE_INT)
				if n == chunks:
					builder.add_arg(
						msg[n * 1024 :], OscMessageBuilder.ARG_TYPE_BLOB
					)
				else:
					builder.add_arg(
						msg[n * 1024 : (n + 1) * 1024],
						OscMessageBuilder.ARG_TYPE_BLOB,
					)
				built = builder.build()
				self._client.send(built)
				self.logger.debug(
					"EngineProxyManager: sent the %d-byte chunk %d of message %d to %s",
					len(built.dgram),
					n,
					self._top_uid,
					built.address,
				)
			self.logger.debug(
				"EngineProxyManager: sent %d bytes",
				len(msg),
			)
			if cmd == "close":
				self.logger.debug("EngineProxyManager: closing input loop")
				return

	def _receive_input(self, _, uid: int, chunks: int, msg: bytes) -> None:
		if uid != self._top_uid:
			self.logger.error(
				"EngineProxyManager: expected uid %d, got uid %d",
				self._top_uid,
				uid,
			)
		self.logger.debug(
			"EngineProxyManager: received %d bytes of the %dth chunk out of %d for uid %d",
			len(msg),
			len(self._input_received),
			chunks,
			uid,
		)
		self._input_received.append(msg)
		if len(self._input_received) == chunks:
			recvd = zlib.decompress(b"".join(self._input_received))
			self.logger.debug(
				"EngineProxyManager: received a complete message, "
				f"decompressed to {len(recvd)} bytes"
			)
			self._input_queue.put(recvd)
			self._top_uid += 1
			self._input_received = []

	def _start_subthread(self, prefix: str | None = None, **kwargs):
		if hasattr(self, "_t"):
			if self._really_shutdown:
				raise RuntimeError("Already started")
			self.logger.info(
				"EngineProxyManager: already have a subthread, will reuse"
			)
			if hasattr(self, "engine_proxy"):
				pack = self.engine_proxy.pack
				unpack = self.engine_proxy.unpack
			else:
				eng = EngineFacade(None)
				pack = eng.pack
				unpack = eng.unpack
			restart_bytes = pack(
				{"command": "restart", "prefix": prefix, **kwargs}
			)
			self._input_queue.put(restart_bytes)
			if not (got := self._output_queue.get()).endswith(
				b"\xa9restarted"
			):
				try:
					gotten = unpack(got)
					if isinstance(gotten, Exception):
						raise gotten
					else:
						raise RuntimeError(
							"Failed to restart subthread", gotten
						)
				except ValueError:
					raise RuntimeError("Failed to restart subthread", got)
			return
		self.logger.debug("EngineProxyManager: starting subthread!")
		from queue import SimpleQueue

		self._input_queue = SimpleQueue()
		self._output_queue = SimpleQueue()

		self._t = Thread(
			target=engine_subthread,
			args=(
				(prefix,),
				kwargs,
				self._input_queue,
				self._output_queue,
				None,
			),
		)
		self._t.start()

	def _start_subinterpreter(self, prefix: str | None = None, **kwargs):
		if hasattr(self, "_terp"):
			if self._really_shutdown:
				raise RuntimeError("Already started")
			self.logger.info(
				"EngineProxyManager: already have a subinterpreter"
			)
			if hasattr(self, "engine_proxy"):
				pack = self.engine_proxy.pack
			else:
				pack = EngineFacade(None).pack
			self._input_queue.put(
				pack({"command": "restart", "prefix": prefix, **kwargs})
			)
			if not (got := self._output_queue.get()).endswith(
				b"\xa9restarted"
			):
				raise RuntimeError("Failed to restart subinterpreter", got)
			return
		from concurrent.interpreters import create, create_queue
		from queue import Queue

		self._input_queue: Queue = create_queue()
		self._output_queue: Queue = create_queue()
		self._logq: Queue = create_queue()

		self._terp = create()
		self._t = self._terp.call_in_thread(
			engine_subthread,
			(prefix,),
			kwargs,
			self._input_queue,
			self._output_queue,
			self._logq,
		)
		self._log_thread = Thread(target=self._sync_log_forever)
		self._log_thread.start()

	def _make_proxy(
		self,
		prefix: Path | None,
		install_modules=(),
		enforce_end_of_time=False,
		game_source_code: dict[str, str] | None = None,
		game_strings: dict[str, str] | None = None,
		**kwargs,
	):
		if hasattr(self, "engine_proxy"):
			raise RuntimeError(
				"Tried to make a second proxy in EngineProxyManager"
			)
		if hasattr(self, "_input_queue"):
			self._input_queue.put(b"echoReadyToMakeProxy")
			if (got := self._output_queue.get()) != b"ReadyToMakeProxy":
				raise RuntimeError("Subthread isn't ready", got)
		else:
			self._proxy_out_pipe.send_bytes(b"echoReadyToMakeProxy")
			if (
				got := self._proxy_in_pipe.recv_bytes()
			) != b"ReadyToMakeProxy":
				raise RuntimeError("Subprocess isn't ready", got)
		branches_d, eternal_d = self._initialize_proxy_db(prefix, **kwargs)
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

		if hasattr(self, "_proxy_in_pipe") and hasattr(
			self, "_proxy_out_pipe"
		):
			self.engine_proxy = EngineProxy(
				self._proxy_in_pipe.recv_bytes,
				self._proxy_out_pipe.send_bytes,
				self.logger,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
				branches_d=branches_d,
				eternal=eternal_d,
				strings=game_strings,
				**game_source_code,
			)
		else:
			self.engine_proxy = EngineProxy(
				self._output_queue.get,
				self._input_queue.put,
				self.logger,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
				branches_d=branches_d,
				eternal=eternal_d,
				strings=game_strings,
				**game_source_code,
			)
			if self.android:
				self._output_sender_thread = Thread(
					target=self._send_output_forever,
					args=[self._output_queue],
				)
				self._output_sender_thread.start()

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
			match self.sub_mode:
				case Sub.interpreter:
					self._start_subinterpreter(prefix, **kwargs)
				case Sub.process:
					self._start_subprocess(prefix, **kwargs)
				case Sub.thread:
					self._start_subthread(prefix, **kwargs)
		pack = EngineFacade(None).pack
		if hasattr(self, "_proxy_out_pipe"):
			self._proxy_out_pipe.send_bytes(
				b"from_archive"
				+ pack(
					{
						"archive_path": str(archive_path),
						"prefix": str(prefix),
						**kwargs,
					}
				)
			)
		else:
			self._input_queue.put(
				b"from_archive"
				+ pack(
					{
						"archive_path": str(archive_path),
						"prefix": str(prefix),
						**kwargs,
					}
				),
			)
		self._make_proxy(prefix, game_source_code=game_code, **kwargs)
		self.engine_proxy._init_pull_from_core()
		return self.engine_proxy

	def close(self):
		self.shutdown()
		if hasattr(self, "_client"):
			self._client.send_message("127.0.0.1/shutdown")
			self.logger.debug(
				"EngineProxyManager: joining input sender thread"
			)
			self._output_sender_thread.join()
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
