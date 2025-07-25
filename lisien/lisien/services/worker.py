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

import os
import random
import sys
import traceback
import zlib
from ast import literal_eval
from functools import partial
from threading import Event, Thread

from kivy.logger import Logger
from pythonosc import osc_server, udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_message_builder import OscMessageBuilder

from lisien.proxy import EngineProxy

Logger.setLevel(0)
Logger.debug("worker: imported libs")


class CommandDispatcher:
	def __init__(
		self, i: int, engine: EngineProxy, client: udp_client.SimpleUDPClient
	):
		self._i = i
		self._engine = engine
		self._client = client
		self._parts = []

	def dispatch_command(self, n: int, inst: bytes):
		self._parts.append(inst)
		if len(self._parts) < n:
			return
		inst = b"".join(self._parts)
		uid = int.from_bytes(inst[:8], "little")
		eng = self._engine
		method, args, kwargs = eng.unpack(zlib.decompress(inst[8:]))
		Logger.debug(
			f"about to dispatch {method} call id {uid} to worker {self._i}"
		)
		if isinstance(method, str):
			method = getattr(eng, method)
		try:
			ret = method(*args, **kwargs)
		except Exception as ex:
			ret = ex
			if uid == sys.maxsize:
				msg = repr(ex)
				eng.critical(msg)
				traceback.print_exc(file=sys.stderr)
				sys.exit(msg)
		eng._initialized = True
		payload = inst[:8] + zlib.compress(eng.pack(ret))
		builder = OscMessageBuilder("/worker-reply")
		builder.add_arg(payload, builder.ARG_TYPE_BLOB)
		self._client.send(builder.build())
		Logger.debug(
			f"sent a reply to call {uid} of method {method}; {len(payload)} bytes"
		)


def worker_server(
	i: int,
	lowest_port: int,
	highest_port: int,
	manager_port: int,
	replies_port: int,
	prefix: str,
	branches: dict,
	eternal: dict,
):
	eng = EngineProxy(
		None,
		None,
		Logger,
		prefix=prefix,
		worker_index=i,
		eternal=eternal,
		branches=branches,
	)
	client = udp_client.SimpleUDPClient("127.0.0.1", replies_port)
	dispatcher = Dispatcher()
	cmddisp = CommandDispatcher(i, eng, client)
	dispatcher.map("/", cmddisp.dispatch_command)
	for _ in range(128):
		my_port = random.randint(lowest_port, highest_port)
		try:
			serv = osc_server.BlockingOSCUDPServer(
				(
					"127.0.0.1",
					my_port,
				),
				dispatcher,
			)
			break
		except OSError:
			continue
	else:
		sys.exit("Couldn't get port for worker %d" % i)

	is_shutdown = Event()

	def shutdown(_, __):
		Logger.debug(f"worker {i}: shutdown called")
		is_shutdown.set()

	udp_client.SimpleUDPClient("127.0.0.1", manager_port).send_message(
		"/worker-report-port", my_port
	)
	dispatcher.map("/shutdown", shutdown)
	Logger.debug(
		"worker %d: Started Lisien worker service %d in prefix %s on port %d. "
		"Sending replies to port %d, and my own port to port %d",
		i,
		i,
		prefix,
		my_port,
		replies_port,
		manager_port,
	)
	return is_shutdown, serv


if __name__ == "__main__":
	try:
		Logger.debug("worker.__main__")
		assert "PYTHON_SERVICE_ARGUMENT" in os.environ
		assert isinstance(os.environ["PYTHON_SERVICE_ARGUMENT"], str)
		args = literal_eval(os.environ["PYTHON_SERVICE_ARGUMENT"])
		Logger.info(f"worker {args[0]}: starting...")
		is_shutdown, serv = worker_server(*args)
		thread = Thread(target=serv.serve_forever)
		thread.start()
		is_shutdown.wait()
		serv.shutdown()
		thread.join()
		Logger.info(f"worker {args[0]}: exited.")
	except BaseException as ex:
		import traceback
		from io import StringIO

		from kivy.logger import Logger

		bogus = StringIO()
		traceback.print_exception(ex, file=bogus)
		for line in bogus.getvalue().split("\n"):
			Logger.error(line)
