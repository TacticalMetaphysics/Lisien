# This file is part Lisien, a framework for life simulation games.
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
import logging
from logging import LogRecord

import msgpack
from queue import SimpleQueue

from itertools import pairwise

from ast import literal_eval
import random
import sys
from functools import partial
from threading import Thread, Event
import os
import zlib

from kivy.logger import Logger
from pythonosc.osc_tcp_server import BlockingOSCTCPServer
from pythonosc.tcp_client import SimpleTCPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

from lisien.proxy import _engine_subroutine_step, _finish_packing
from lisien.proxy.handle import EngineHandle


Logger.setLevel(0)
Logger.debug("core: imported libs")


def pack4send(hand: EngineHandle, addr: str, o) -> OscMessage:
	builder = OscMessageBuilder(addr)
	builder.add_arg(zlib.compress(hand.pack(o)), builder.ARG_TYPE_BLOB)
	return builder.build()


class CommandDispatcher:
	def __init__(self, handle: EngineHandle, client: SimpleTCPClient):
		self._handle = handle
		self._client = client
		self._parts = []

	def dispatch_command(self, _, chunks: int, inst: bytes):
		Logger.debug(
			"core: in dispatch_command, got %d bytes of a %d part message",
			len(inst),
			chunks,
		)
		self._parts.append(inst)
		if len(self._parts) < chunks:
			return

		hand = self._handle
		instruction = hand.unpack(zlib.decompress(b"".join(self._parts)))
		Logger.debug(
			f"core: in dispatch_command, collected command "
			f"{instruction.get('command', '???')}"
		)
		self._parts = []

		outq = SimpleQueue()

		def send_output_bytes(cmd, resp: bytes):
			outq.put(
				zlib.compress(
					_finish_packing(hand.pack, cmd, *hand._real._btt(), resp)
				)
			)

		def send_output(cmd, resp):
			send_output_bytes(cmd, hand.pack(resp))

		cmd = instruction.get("command", "nothing??")

		Logger.debug("core: about to dispatch %s to the Lisien core", cmd)

		_engine_subroutine_step(
			hand, instruction, send_output, send_output_bytes
		)

		res = outq.get()
		if len(res) < 1024:
			chunks = [res]
		else:
			chunks = []
			for i, j in pairwise(range(0, len(res), 1024)):
				chunks.append(res[i:j])
			if j < len(res):
				chunks.append(res[j:])

		for chunk in chunks:
			if not isinstance(chunk, bytes):
				raise TypeError("Bad chunk", type(chunk))
			builder = OscMessageBuilder("/")
			builder.add_arg(len(chunks), builder.ARG_TYPE_INT)
			builder.add_arg(chunk, builder.ARG_TYPE_BLOB)
			self._client.send(builder.build())
			Logger.debug(
				"core: replied to %s with %d bytes in %d chunks",
				cmd,
				sum(map(len, chunks)),
				len(chunks),
			)


class CoreLogHandler(logging.Handler):
	def __init__(
		self, pack: callable, client: SimpleTCPClient, level: int = 0
	):
		self._pack = pack
		self._client = client
		super().__init__(level)

	def emit(self, record: LogRecord) -> None:
		builder = OscMessageBuilder("/log")
		builder.add_arg(self._pack(record.__dict__), builder.ARG_TYPE_BLOB)
		self._client.send(builder.build())


def core_server(
	lowest_port: int,
	highest_port: int,
	replies_port: int,
	args: list,
	kwargs: dict,
):
	dispatcher = Dispatcher()
	for _ in range(128):
		my_port = random.randint(lowest_port, highest_port)
		try:
			serv = BlockingOSCTCPServer(
				("127.0.0.1", my_port),
				dispatcher,
			)
			break
		except OSError:
			pass
	else:
		sys.exit("couldn't get core port")
	client = SimpleTCPClient("127.0.0.1", replies_port)
	Logger.debug(
		"core: got port %d, sending it to 127.0.0.1/core-report-port:%d",
		my_port,
		replies_port,
	)
	client.send_message("/core-report-port", my_port)

	logger = logging.getLogger("lisien")

	def pack(obj):
		return hand._real.pack(obj)

	logger.addHandler(CoreLogHandler(pack, client, 0))

	hand = EngineHandle(
		*args,
		port=my_port,
		logger=logger,
		**kwargs,
	)

	is_shutdown = Event()

	def shutdown(_, __):
		Logger.debug("core: shutdown called")
		is_shutdown.set()

	cmddisp = CommandDispatcher(hand, client)

	dispatcher.map("/", cmddisp.dispatch_command)
	dispatcher.map("/shutdown", shutdown)
	dispatcher.map("/connect-workers", hand._real._connect_worker_services)
	Logger.info(
		"core: about to start server at port %d, sending replies to port %d",
		my_port,
		replies_port,
	)
	return is_shutdown, serv


if __name__ == "__main__":
	Logger.info("Starting Lisien core service...")
	args = literal_eval(os.environ["PYTHON_SERVICE_ARGUMENT"])
	is_shutdown, serv = core_server(*args)
	thread = Thread(target=serv.serve_forever)
	thread.start()
	is_shutdown.wait()
	Logger.debug("core: about to call serv.shutdown")
	serv.shutdown()
	Logger.debug("core: serv.shutdown worked")
	thread.join()
	Logger.info("core: Lisien core service has ended")
