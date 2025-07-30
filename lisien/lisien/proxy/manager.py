from __future__ import annotations

import logging
import os
import pickle
import random
import sys
from threading import Thread

import msgpack
import tblib

from .engine import EngineProxy
from .process import engine_subprocess, engine_subthread


class EngineProcessManager:
	"""Container for a Lisien proxy and a logger for it

	Make sure the :class:`EngineProcessManager` instance lasts as long as the
	:class:`lisien.proxy.EngineProxy` returned from its :method:`start`
	method. Call the :method:`EngineProcessManager.shutdown` method
	when you're done with the :class:`lisien.proxy.EngineProxy`. That way,
	we can join the thread that listens to the subprocess's logs.

	:param use_thread: Actually use a thread, not a process.

	"""

	loglevel = logging.DEBUG

	def __init__(
		self,
		*args,
		use_thread: bool = False,
		service_class_name: str = "org.tacmeta.elide.core",
		**kwargs,
	):
		self.use_thread = use_thread
		self.service_class_name = service_class_name
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
		if hasattr(self, "_handle_in_pipe") and hasattr(
			self, "_handle_out_pipe"
		):
			from multiprocessing import Process

			self._p = Process(
				name="Lisien Life Simulator Engine (core)",
				target=engine_subprocess,
				args=(
					args or self._args,
					self._kwargs | kwargs,
					self._handle_in_pipe,
					self._handle_out_pipe,
				),
				kwargs={"log_queue": self._logq},
			)
			self._p.start()
			self._log_thread = Thread(
				target=self._sync_log_forever, daemon=True
			)
			self._log_thread.start()
			self.engine_proxy = EngineProxy(
				self._proxy_in_pipe.recv,
				self._proxy_out_pipe.send,
				self.logger,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
			)
		elif android:
			from queue import SimpleQueue

			from jnius import autoclass
			from pythonosc.dispatcher import Dispatcher
			from pythonosc.osc_message_builder import OscMessageBuilder
			from pythonosc.osc_tcp_server import ThreadingOSCTCPServer
			from pythonosc.tcp_client import SimpleTCPClient

			if "workers" in kwargs:
				workers = kwargs["workers"]
			else:
				workers = os.cpu_count()
			if "logger" in kwargs:
				raise ValueError(
					"Can't pass loggers between processes on Android"
				)
			# Android makes us hardcode some number of service workers, to be
			# used or not. I've defined 64 of them in buildozer.spec.
			workers = min((workers, 64))
			self._output_queue = output_queue = SimpleQueue()
			self._input_queue = input_queue = SimpleQueue()
			worker_port_queue = SimpleQueue()
			core_port_queue = SimpleQueue()
			disp = Dispatcher()
			disp.map(
				"/core-report-port", lambda _, port: core_port_queue.put(port)
			)
			disp.map(
				"/worker-report-port",
				lambda _, port: worker_port_queue.put(port),
			)
			disp.map("/log", self._handle_log_record)
			self._output_received = []
			disp.map("/", self._receive_output)
			low_port = 32000
			high_port = 60999
			for _ in range(128):
				procman_port = random.randint(low_port, high_port)
				try:
					self._server = ThreadingOSCTCPServer(
						("127.0.0.1", procman_port), disp
					)
					self._server_thread = Thread(
						target=self._server.serve_forever
					)
					self._server_thread.start()
					self.logger.debug(
						"EngineProcessManager: started server at port %d",
						procman_port,
					)
					break
				except OSError:
					pass
			else:
				sys.exit("couldn't get port for process manager")

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
					ParquetDB("global")
					.read(columns=["key", "value"])
					.to_pylist()
				):
					eternal_d[d["key"]] = d["value"]
			self.engine_proxy = EngineProxy(
				input_queue,
				output_queue,
				self.logger,
				install_modules,
				replay_file=replay_file,
				branches=branches_d,
				eternal=eternal_d,
			)
			self.logger.debug("EngineProcessManager: made engine proxy")
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
					kwargs | self._kwargs | {"workers": workers},
				]
			)
			try:
				core_service.start(mActivity, argument)
			except Exception as ex:
				self.logger.critical(repr(ex))
				sys.exit(repr(ex))
			self.logger.debug("EngineProcessManager: started core")
			core_port = core_port_queue.get()
			self._client = SimpleTCPClient("127.0.0.1", core_port)
			self.logger.debug(
				"EngineProcessManager: connected to lisien core at port %d",
				core_port,
			)
			if workers:
				for i in range(workers):
					argument = repr(
						[
							i,
							low_port,
							high_port,
							procman_port,
							core_port,
							args[0],
							branches_d,
							eternal_d,
						]
					)
					autoclass(f"org.tacmeta.elide.ServiceWorker{i}").start(
						mActivity, argument
					)
					self.logger.debug(
						"EngineProcessManager: started worker %d", i
					)
				worker_ports = []
				for i in range(workers):
					port = worker_port_queue.get()
					self.logger.debug(
						"EngineProcessManager: worker %d says it's on port %d",
						i,
						port,
					)
					worker_ports.append(port)
				workers_payload = OscMessageBuilder("/connect-workers")
				workers_payload.add_arg(
					msgpack.packb(worker_ports),
					OscMessageBuilder.ARG_TYPE_BLOB,
				)
				self._client.send(workers_payload.build())
				self.logger.debug(
					"EngineProcessManager: sent ports to core/connect-workers"
				)
			self._top_uid = 0
			self._input_sender_thread = Thread(
				target=self._send_input_forever,
				args=[input_queue],
				daemon=True,
			)
			self._input_sender_thread.start()
		else:
			raise RuntimeError("Couldn't start process or service")

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

	def _send_input_forever(self, input_queue):
		from pythonosc.osc_message_builder import OscMessageBuilder

		assert hasattr(self, "engine_proxy"), (
			"EngineProcessManager tried to send input with no EngineProxy"
		)
		while True:
			cmd = input_queue.get()
			msg = self.engine_proxy.pack(cmd)
			chunks = len(msg) // 1024
			if len(msg) % 1024:
				chunks += 1
			self.logger.debug(
				f"EngineProcessManager: about to send {cmd} to core in {chunks} chunks"
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
					"EngineProcessManager: sent the %d-byte chunk %d of message %d to %s",
					len(built.dgram),
					n,
					self._top_uid,
					built.address,
				)
			self.logger.debug(
				"EngineProcessManager: sent %d bytes of %s",
				len(msg),
				cmd.get("command", "???"),
			)
			if cmd == "close":
				self.logger.debug("EngineProcessManager: closing input loop")
				return

	def _receive_output(self, _, uid: int, chunks: int, msg: bytes) -> None:
		if uid != self._top_uid:
			self.logger.error(
				"EngineProcessManager: expected uid %d, got uid %d",
				self._top_uid,
				uid,
			)
		self.logger.debug(
			"EngineProcessManager: received %d bytes of the %dth chunk out of %d for uid %d",
			len(msg),
			len(self._output_received),
			chunks,
			uid,
		)
		self._output_received.append(msg)
		if len(self._output_received) == chunks:
			self._output_queue.put(
				self.engine_proxy.unpack(b"".join(self._output_received))
			)
			self._top_uid += 1
			self._output_received = []

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
				self.engine_proxy._pipe_out.send_bytes(b"shutdown")
				self._p.join(timeout=1)
			del self.engine_proxy
		if hasattr(self, "_client"):
			self._client.send_message("/shutdown", "")
			self._core_service.stop(self._mActivity)
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
		try:
			import android
		except ImportError:
			if not hasattr(self, "use_thread"):
				from multiprocessing import Pipe, SimpleQueue

				(self._handle_in_pipe, self._proxy_out_pipe) = Pipe(
					duplex=False
				)
				(self._proxy_in_pipe, self._handle_out_pipe) = Pipe(
					duplex=False
				)
				self._logq = SimpleQueue()
		if hasattr(self, "_handle_in_pipe") and hasattr(
			self, "_handle_out_pipe"
		):
			from multiprocessing import Process

			self._p = Process(
				name="Lisien Life Simulator Engine (core)",
				target=engine_subprocess,
				args=(
					args or self._args,
					self._kwargs | kwargs,
					self._handle_in_pipe,
					self._handle_out_pipe,
				),
				kwargs={"log_queue": self._logq},
			)
			self._p.start()
		else:
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

	def _make_proxy(self, *args, **kwargs):
		self._config_logger(kwargs)
		install_modules = (
			kwargs.pop("install_modules")
			if "install_modules" in kwargs
			else []
		)
		if not hasattr(self, "_t"):
			self._start_subprocess(*args, **kwargs)
		enforce_end_of_time = kwargs.get("enforce_end_of_time", False)
		if hasattr(self, "_handle_in_pipe") and hasattr(
			self, "_handle_out_pipe"
		):
			self.engine_proxy = EngineProxy(
				self._proxy_in_pipe,
				self._proxy_out_pipe,
				self.logger,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
			)
		else:
			self.engine_proxy = EngineProxy(
				self._input_queue,
				self._output_queue,
				self.logger,
				install_modules,
				enforce_end_of_time=enforce_end_of_time,
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
