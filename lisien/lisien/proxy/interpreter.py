def worker_subinterpreter(
	sys_path,
	i,
	prefix: str,
	branches: dict,
	eternal: dict,
	in_queue,
	out_queue,
	logq,
):
	import sys
	import logging

	sys.path.extend(sys_path)

	from lisien.proxy.engine import (
		EngineProxy,
		WorkerLogHandler,
	)

	logger = logging.getLogger(f"lisien worker {i}")
	handler = WorkerLogHandler(logq, logging.DEBUG, i)
	logger.addHandler(handler)
	eng = EngineProxy(
		None,
		None,
		logger,
		prefix=prefix,
		worker_index=i,
		eternal=eternal,
		branches=branches,
	)
	pack = eng.pack
	unpack = eng.unpack
	eng._initialized = False
	get_input_bytes = in_queue.get
	send_output_bytes = out_queue.put
	while True:
		inst = get_input_bytes()
		if inst == b"shutdown":
			if hasattr(in_queue, "close"):
				in_queue.close()
			if logq and hasattr(logq, "close"):
				logq.close()
			send_output_bytes(b"done")
			if hasattr(out_queue, "close"):
				out_queue.close()
			return
		uid = int.from_bytes(inst[:8], "little")
		(method, args, kwargs) = unpack(inst[8:])
		if isinstance(method, str):
			method = getattr(eng, method)
		try:
			ret = method(*args, **kwargs)
		except Exception as ex:
			ret = ex
			if uid == sys.maxsize:
				msg = repr(ex)
				logq.put((50, msg))
				import traceback

				traceback.print_exc(file=sys.stderr)
				sys.exit(msg)
		if uid != sys.maxsize:
			send_output_bytes(inst[:8] + pack(ret))
		eng._initialized = True
