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
from enum import Enum
from threading import Thread
from zipfile import ZipFile

import tblib

from ..types import Branch, EternalKey, Key, Tick, Turn, Value
from .engine import EngineProxy
from .routine import engine_subprocess, engine_subthread


class Sub(Enum):
	process = "process"
	interpreter = "interpreter"
	thread = "thread"


GET_TIME = b"\x81\xa7command\xa8get_time"


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

	def __init__(
		self,
		*args,
		sub_mode: Sub = Sub.thread,
		**kwargs,
	):
		self.sub_mode = Sub(sub_mode)
		self._args = args
		self._kwargs = kwargs
		self._top_uid = 0

	def start(self, *args, **kwargs):
		"""Start lisien in a subprocess, and return a proxy to it"""
		if hasattr(self, "engine_proxy"):
			raise RuntimeError("Already started")

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
		args = args or self._args
		kwargs |= self._kwargs
		if args and "prefix" in kwargs:
			raise TypeError(
				"Got multiple arguments for prefix", args[0], kwargs["prefix"]
			)
		elif args:
			prefix = args[0]
		elif "prefix" in kwargs:
			prefix = kwargs.pop("prefix")
		else:
			raise RuntimeError("No prefix")
		self._make_proxy(prefix, **kwargs)
		if hasattr(self, "_proxy_out_pipe"):
			self._proxy_out_pipe.send_bytes(GET_TIME)
		else:
			self._output_queue.put(GET_TIME)
		self.engine_proxy._init_pull_from_core()
		return self.engine_proxy

	def _sync_log_forever(self):
		while True:
			logrec = self._logq.get()
			if logrec == b"shutdown":
				return
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
		self, prefix, **kwargs
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

		if "connect_string" in kwargs:
			from sqlalchemy import NullPool, create_engine, select
			from sqlalchemy.exc import OperationalError

			from ..sql import meta

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
		elif prefix is None:
			self.logger.warning(
				"Running without a database. Lisien will be empty at start."
			)
		else:
			from parquetdb import ParquetDB

			pqdb_prefix = os.path.join(prefix, "world")

			for d in (
				ParquetDB(f"{pqdb_prefix}/branches")
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
			self.engine_proxy.send_bytes(b"shutdown")
			if hasattr(self, "_p"):
				self._p.join(timeout=1)
			del self.engine_proxy
		if hasattr(self, "_client"):
			while not self._output_queue.empty():
				time.sleep(0.01)
		if hasattr(self, "_server"):
			self._server.shutdown()
		if hasattr(self, "_logq"):
			self._logq.put(b"shutdown")
			self._log_thread.join()
		if hasattr(self, "_t"):
			self._t.join(timeout=1.0)
			if self._t.is_alive():
				raise TimeoutError("Couldn't join thread")
		if hasattr(self, "_terp_thread") and self._terp_thread.is_alive():
			self._terp_thread.join(timeout=1.0)
			if self._terp_thread.is_alive():
				raise TimeoutError("Couldn't join interpreter thread")
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
				self._logq,
			),
		)
		self._p.start()

		self._log_thread = Thread(target=self._sync_log_forever, daemon=True)
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

	def _start_subthread(self, *args, **kwargs):
		self.logger.debug("EngineProxyManager: starting subthread!")
		from queue import SimpleQueue

		self._input_queue = SimpleQueue()
		self._output_queue = SimpleQueue()

		self._t = Thread(
			target=engine_subthread,
			args=(
				args or self._args,
				self._kwargs | kwargs,
				self._output_queue,
				self._input_queue,
				None,
			),
		)
		self._t.start()

	def _start_subinterpreter(self, *args, **kwargs):
		from concurrent.interpreters import create, create_queue
		from queue import Queue

		self._input_queue: Queue = create_queue()
		self._output_queue: Queue = create_queue()
		self._logq: Queue = create_queue()

		self._terp = create()
		self._terp_thread = self._terp.call_in_thread(
			engine_subthread,
			args or self._args,
			self._kwargs | kwargs,
			self._output_queue,
			self._input_queue,
			self._logq,
		)
		self._log_thread = Thread(target=self._sync_log_forever, daemon=True)
		self._log_thread.start()

	def _make_proxy(
		self,
		prefix,
		install_modules=(),
		enforce_end_of_time=False,
		game_source_code: dict[str, str] | None = None,
		game_strings: dict[str, str] | None = None,
		**kwargs,
	):
		branches_d, eternal_d = self._initialize_proxy_db(prefix, **kwargs)
		if game_source_code is None:
			game_source_code = {}
			if prefix is not None:
				for store in (
					"function",
					"method",
					"trigger",
					"prereq",
					"action",
				):
					pyfile = os.path.join(prefix, store + ".py")
					if os.path.exists(pyfile) and os.stat(pyfile).st_size:
						code = game_source_code[store] = {}
						with open(pyfile, "rt") as inf:
							parsed = ast.parse(inf.read(), pyfile)
						funk: ast.FunctionDef
						for funk in parsed.body:
							code[funk.name] = ast.unparse(funk)
		if game_strings is None:
			if prefix and os.path.isdir(os.path.join(prefix, "strings")):
				lang = eternal_d.get(EternalKey(Key("language")), "eng")
				jsonpath = os.path.join(prefix, "strings", str(lang) + ".json")
				if os.path.isfile(jsonpath):
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
				branches=branches_d,
				eternal=eternal_d,
				strings=game_strings,
				**game_source_code,
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
				strings=game_strings,
				**game_source_code,
			)
			if self.android:
				self._output_sender_thread = Thread(
					target=self._send_output_forever,
					args=[self._output_queue],
					daemon=True,
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
		if not archive_path.endswith(".lisien"):
			raise RuntimeError("Not a .lisien archive")
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
		try:
			from msgpack import Packer

			if Packer.__module__.endswith("cmsgpack"):
				from msgpack import packb
			else:
				from umsgpack import packb
		except ImportError:
			from umsgpack import packb
		if hasattr(self, "_proxy_out_pipe"):
			self._proxy_out_pipe.send_bytes(
				b"from_archive"
				+ packb(
					{"archive_path": archive_path, "prefix": prefix, **kwargs}
				)
			)
		else:
			self._output_queue.put(
				(
					"from_archive",
					{"archive_path": archive_path, "prefix": prefix, **kwargs},
				)
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
		self.logger.debug("EngineProxyManager: closed")

	def __enter__(self):
		return self.start()

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()
