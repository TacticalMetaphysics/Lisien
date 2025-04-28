import base64
from functools import partial
from logging import getLogger
import os
import sys
import traceback
import zlib

import msgpack
from pythonosc import osc_tcp_server
from pythonosc import tcp_client
from pythonosc.dispatcher import Dispatcher

from lisien.proxy import EngineProxy


def dispatch_command(
	eng: EngineProxy, client: tcp_client.SimpleTCPClient, inst: bytes
):
	uid = int.from_bytes(inst[:8], "little")
	method, args, kwargs = eng.unpack(zlib.decompress(inst[8:]))
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


def worker_service(
	i: int,
	my_port: int,
	replies_port: int,
	prefix: str,
	branches: dict,
	eternal: dict,
):
	logger = getLogger(f"lisien_worker_{my_port}")
	logger.info(
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
	client = tcp_client.SimpleTCPClient("127.0.0.1", replies_port)
	dispatcher = Dispatcher()
	dispatcher.map("/", partial(dispatch_command, eng, client))
	serv = osc_tcp_server.BlockingOSCTCPServer(
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
