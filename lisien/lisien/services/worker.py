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

import base64
from functools import partial
import os
import sys
import traceback
import zlib

from kivy.logger import Logger
import msgpack
from pythonosc import osc_server
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher

from lisien.proxy import EngineProxy


Logger.debug("worker: imported libs")


def dispatch_command(
	i, eng: EngineProxy, client: udp_client.SimpleUDPClient, inst: bytes
):
	uid = int.from_bytes(inst[:8], "little")
	method, args, kwargs = eng.unpack(zlib.decompress(inst[8:]))
	Logger.debug(f"about to dispatch {method} call id {uid} to worker {i}")
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
	client.send_message("/worker-reply", payload)
	Logger.debug(
		f"sent a reply to call {uid} of method {method}; {len(payload)} bytes"
	)


def worker_service(
	i: int,
	my_port: int,
	replies_port: int,
	prefix: str,
	branches: dict,
	eternal: dict,
):
	Logger.debug(
		"Started Lisien worker service in prefix %s on port %d, "
		"sending replies to port %d",
		prefix,
		my_port,
		replies_port,
	)

	eng = EngineProxy(
		None,
		None,
		logger,
		prefix=prefix,
		worker_index=i,
		eternal=eternal,
		branches=branches,
	)
	client = udp_client.SimpleUDPClient("127.0.0.1", replies_port)
	dispatcher = Dispatcher()
	dispatcher.map("/", partial(dispatch_command, i, eng, client))
	serv = osc_server.BlockingOSCUDPServer(
		(
			"127.0.0.1",
			my_port,
		),
		dispatcher,
	)
	dispatcher.map("/shutdown", lambda _: serv.shutdown())
	serv.serve_forever()


if __name__ == "__main__":
	assert "PYTHON_SERVICE_ARGUMENT" in os.environ
	assert isinstance(os.environ["PYTHON_SERVICE_ARGUMENT"], str)
	worker_service(
		*msgpack.unpackb(
			zlib.decompress(
				base64.urlsafe_b64decode(os.environ["PYTHON_SERVICE_ARGUMENT"])
			)
		)
	)
