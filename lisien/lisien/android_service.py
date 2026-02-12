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
"""The background process that Lisien runs in on Android.

It's a tiny `OSC`_ server. All it does
is start the same :class:`lisien.proxy.handle.EngineHandle` that's used
for accessing an :class:`lisien.engine.Engine` in a subprocess, but,
since you're not allowed to start subprocesses on Android, we have to do it
in a server instead.

It's probably a bad idea to use this anywhere outside of ``localhost``, since
:mod:`lisien.android_service` performs no authentication whatsoever.

If using Kivy's `Buildozer`_ tool to package an Elide-based app, make sure that
your buildozer.spec file includes the line:

``services =  core:_python_bundle/site-packages/lisien/android_service.py``

This module will run in the background and accept commands from Elide via
`OSC`_. There are only two public
endpoints: ``/`` accepts packed instructions, possibly split into chunks;
``/shutdown`` commits pending changes and ends the server process.

.. _Buildozer: https://buildozer.readthedocs.io/en/latest/
.. _OSC: https://opensoundcontrol.stanford.edu/

"""

import logging
import os
import pickle
import random
import sys
import zlib
from ast import literal_eval
from itertools import pairwise
from logging import LogRecord
from queue import SimpleQueue
from threading import Event, Thread
from typing import Callable

import tblib
from kivy.logger import Logger
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_tcp_server import BlockingOSCTCPServer
from pythonosc.tcp_client import SimpleTCPClient

from lisien.proxy.handle import EngineHandle
from lisien.proxy.routine import _engine_subroutine_step, _finish_packing

Logger.setLevel(0)
Logger.debug("core: imported libs")


class CommandDispatcher:
	"""Glue between OSC and :class:`lisien.proxy.handle.EngineHandle`"""

	def __init__(self, handle: EngineHandle, client: SimpleTCPClient):
		self._handle = handle
		self._client = client
		self._parts = []
		self._last_uid = 0

	def dispatch_command(self, _, uid: int, chunks: int, inst: bytes):
		"""Collect commands in multiple chunks and do what they say

		:param uid: An arbitrary integer that should be unique per run.
		:param chunks: How many chunks the whole message takes up.
		Each chunk should be no more than a kilobyte large.
		:param inst: Bytes encoding part of the instruction.

		"""
		Logger.debug(
			"core: in dispatch_command, got %d bytes of a %d part message with uid %d",
			len(inst),
			chunks,
			uid,
		)
		if self._parts and uid != self._last_uid:
			Logger.error(
				"core: expected uid %d, got uid %d", self._last_uid, uid
			)
		self._last_uid = uid
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
					_finish_packing(hand.pack, cmd, *hand._real.time, resp)
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
			j = -1
			for i, j in pairwise(range(0, len(res), 1024)):
				chunks.append(res[i:j])
			if j < len(res):
				chunks.append(res[j:])

		for chunk in chunks:
			if not isinstance(chunk, bytes):
				raise TypeError("Bad chunk", type(chunk))
			builder = OscMessageBuilder("/")
			builder.add_arg(self._last_uid, builder.ARG_TYPE_INT)
			builder.add_arg(len(chunks), builder.ARG_TYPE_INT)
			builder.add_arg(chunk, builder.ARG_TYPE_BLOB)
			self._client.send(builder.build())
			Logger.debug(
				"core: replied to %s with %d bytes in %d chunks",
				cmd,
				sum(map(len, chunks)),
				len(chunks),
			)
		if cmd == "close":
			self._client.close()


class CoreLogHandler(logging.Handler):
	"""Send all log records to Elide, pickled"""

	def __init__(
		self, pack: Callable, client: SimpleTCPClient, level: int = 0
	):
		self._pack = pack
		self._client = client
		super().__init__(level)

	def emit(self, record: LogRecord) -> None:
		builder = OscMessageBuilder("/log")
		if record.exc_info:
			if (
				isinstance(record.exc_info, Exception)
				and record.exc_info.__traceback__
			):
				record.exc_info.__traceback__ = tblib.Traceback(
					record.exc_info.__traceback__
				).as_dict()
			elif (
				isinstance(record.exc_info, tuple)
				and len(record.exc_info) == 3
				and record.exc_info[2]
			):
				record.exc_info = (
					record.exc_info[0],
					record.exc_info[1],
					tblib.Traceback(record.exc_info[2]).as_dict(),
				)
		builder.add_arg(pickle.dumps(record), builder.ARG_TYPE_BLOB)
		self._client.send(builder.build())


def core_server(
	lowest_port: int,
	highest_port: int,
	replies_port: int,
	args: list,
	kwargs: dict,
):
	"""The OSC server process

	:param lowest_port: The lowest port to try serving on
	:param highest_port: The highest port to try serving on
	:param replies_port: The port to send the results of commands to
	:param args: Positional arguments to :class:`lisien.proxy.handle.EngineHandle`
	:param kwargs: Keyword arguments to :class:`lisien.proxy.handle.EngineHandle`

	"""
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
		logger=logger,
		**kwargs,
	)
	hand.debug("started engine handle in core_server")

	is_shutdown = Event()

	def shutdown(_):
		Logger.debug("core: shutdown called")
		is_shutdown.set()

	cmddisp = CommandDispatcher(hand, client)

	dispatcher.map("/", cmddisp.dispatch_command)
	dispatcher.map("/shutdown", shutdown)
	Logger.info(
		"core: about to start server at port %d, sending replies to port %d",
		my_port,
		replies_port,
	)
	return is_shutdown, serv


if __name__ == "__main__":
	try:
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
	except BaseException as ex:
		import traceback
		from io import StringIO

		from kivy.logger import Logger

		bogus = StringIO()
		traceback.print_exception(ex, file=bogus)
		for line in bogus.getvalue().split("\n"):
			Logger.error(line)
