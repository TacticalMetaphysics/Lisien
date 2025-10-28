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
