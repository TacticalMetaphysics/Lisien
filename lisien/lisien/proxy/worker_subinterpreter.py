def worker_subinterpreter(
	i: int,
	prefix: str,
	branches: dict,
	eternal: dict,
	in_queue,
	out_queue,
	logq,
	*,
	function: dict | None,
	method: dict | None,
	trigger: dict | None,
	prereq: dict | None,
	action: dict | None,
):
	from lisien.proxy.routine import worker_subroutine

	return worker_subroutine(
		i,
		prefix,
		branches,
		eternal,
		in_queue.get,
		out_queue.put,
		logq,
		function=function,
		method=method,
		trigger=trigger,
		prereq=prereq,
		action=action,
	)
