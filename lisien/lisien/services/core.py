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

import base64
from functools import partial
from logging import getLogger, DEBUG
import os
import zlib

import msgpack
from pythonosc import osc_server
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher

from lisien.proxy import _engine_subroutine_step
from lisien.proxy.handle import EngineHandle


logger = getLogger(__name__)
logger.setLevel(DEBUG)


def dispatch_command(
	hand: EngineHandle, client: udp_client.SimpleUDPClient, inst: bytes
):
	instruction = hand.unpack(zlib.decompress(inst))

	def send_output(r):
		client.send_message("/core-reply", zlib.compress(hand.pack(r)))

	def send_output_bytes(resp: bytes):
		client.send_message("/core-reply", zlib.compress(resp))

	_engine_subroutine_step(hand, instruction, send_output, send_output_bytes)


def core_service(my_port: int, replies_port: int, args: list, kwargs: dict):
	from android.permissions import request_permissions, Permission

	request_permissions([Permission.INTERNET])
	client = udp_client.SimpleUDPClient("127.0.0.1", replies_port)
	hand = EngineHandle(
		*args,
		logfun=lambda lvl, msg: client.send_message("/log", [lvl, msg]),
		**kwargs,
	)

	dispatcher = Dispatcher()
	dispatcher.map("/", partial(dispatch_command, hand, client))
	serv = osc_server.BlockingOSCUDPServer(
		("127.0.0.1", my_port),
		dispatcher,
	)
	dispatcher.map("/shutdown", lambda _: serv.shutdown())
	serv.serve_forever()


if __name__ == "__main__":
	logger.info("core.py __main__ executing")
	assert "PYTHON_SERVICE_ARGUMENT" in os.environ
	assert isinstance(os.environ["PYTHON_SERVICE_ARGUMENT"], str)

	core_service(
		*msgpack.unpackb(
			zlib.decompress(
				base64.urlsafe_b64decode(os.environ["PYTHON_SERVICE_ARGUMENT"])
			)
		)
	)
