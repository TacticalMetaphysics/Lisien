import base64
from functools import partial
from logging import getLogger, Logger
import os
from threading import Thread
from queue import SimpleQueue
import sys
import traceback
import zlib

import msgpack
from pythonosc import osc_tcp_server
from pythonosc import tcp_client
from pythonosc.dispatcher import Dispatcher

from lisien.proxy.handle import EngineHandle

from lisien.exc import OutOfTimelineError

from lisien.util import MsgpackExtensionType


def dispatch_command(
	hand: EngineHandle, client: tcp_client.SimpleTCPClient, inst: bytes
):
	instruction = hand.unpack(zlib.decompress(inst))
	silent = instruction.pop("silent", False)
	cmd = instruction.pop("command")
	branching = instruction.pop("branching", False)

	try:
		if branching:
			try:
				r = getattr(hand, cmd)(**instruction)
			except OutOfTimelineError:
				hand.increment_branch()
				r = getattr(hand, cmd)(**instruction)
		else:
			r = getattr(hand, cmd)(**instruction)
	except AssertionError:
		raise
	except Exception as ex:
		client.send_message(
			"/core-reply",
			zlib.compress(
				hand.pack(
					(
						cmd,
						hand._real.branch,
						hand._real.turn,
						hand._real.tick,
						ex,
					)
				)
			),
		)
		return
	if silent:
		return

	resp = msgpack.Packer().pack_array_header(5) + (
		hand.pack(cmd)
		+ hand.pack(hand._real.branch)
		+ hand.pack(hand._real.turn)
		+ hand.pack(hand._real.tick)
	)
	if hasattr(getattr(hand, cmd), "prepacked"):
		if isinstance(r, dict):
			resp += msgpack.Packer().pack_map_header(len(r))
			for k, v in r.items():
				resp += k + v
		elif isinstance(r, tuple):
			pacr = msgpack.Packer()
			pacr.pack_ext_type(
				MsgpackExtensionType.tuple.value,
				msgpack.Packer().pack_array_header(len(r)) + b"".join(r),
			)
			resp += pacr.bytes()
		elif isinstance(r, list):
			resp += msgpack.Packer().pack_array_header(len(r)) + b"".join(r)
		else:
			resp += r
	else:
		resp += hand.pack(r)
	client.send_message("/core-reply", zlib.compress(resp))
	if hasattr(hand, "_after_ret"):
		hand._after_ret()
		del hand._after_ret


def core_service(my_port: int, replies_port: int, args: list, kwargs: dict):
	client = tcp_client.SimpleTCPClient("127.0.0.1", replies_port)
	hand = EngineHandle(
		*args,
		logfun=lambda lvl, msg: client.send_message("/log", [lvl, msg]),
		**kwargs,
	)

	dispatcher = Dispatcher()
	dispatcher.map("/", partial(dispatch_command, hand, client))
	serv = osc_tcp_server.BlockingOSCTCPServer(
		("127.0.0.1", my_port),
		dispatcher,
	)
	dispatcher.map("/shutdown", lambda _: serv.shutdown())
	serv.serve_forever()


if __name__ == "__main__":
	assert "PYTHON_SERVICE_ARGUMENT" in os.environ
	assert isinstance(os.environ["PYTHON_SERVICE_ARGUMENT"], str)

	core_service(
		*msgpack.unpackb(
			zlib.decompress(
				base64.urlsafe_b64decode(os.environ["PYTHON_SERVICE_ARGUMENT"])
			)
		)
	)
