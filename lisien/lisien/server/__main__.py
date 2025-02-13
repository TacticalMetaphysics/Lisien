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
"""A simple local server providing access to a lisien core

Run this as:

	python3 -m lisien.server

and it will start an HTTP server at localhost:8080. Send msgpack mappings
to it, with the key 'command' set to the name of one of the methods in
:class:`lisien.handle.EngineHandle` and remaining keys set to arguments
accepted by that method.

Refer to :class:`lisien.handle.EngineHandle` for documentation on those
methods.

"""

from argparse import ArgumentParser

import cherrypy

from . import LiSEHandleWebService

parser = ArgumentParser()
parser.add_argument("--prefix", action="store", default=".")
args = parser.parse_args()
conf = {
	"/": {
		"request.dispatch": cherrypy.dispatch.MethodDispatcher(),
		"tools.sessions.on": True,
		"tools.response_headers.on": True,
		"tools.response_headers.headers": [
			("Content-Type", "application/json")
		],
		"tools.encode.on": True,
		"tools.encode.encoding": "utf-8",
	}
}
cherrypy.quickstart(LiSEHandleWebService(args.prefix), "/", conf)
