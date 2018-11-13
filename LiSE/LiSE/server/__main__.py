# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import cherrypy
from argparse import ArgumentParser
from . import LiSEHandleWebService

parser = ArgumentParser()
parser.add_argument('world', action='store', required=True)
parser.add_argument('-c', '--code', action='store')
args = parser.parse_args()
conf = {
    '/': {
        'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
        'tools.sessions.on': True,
        'tools.response_headers.on': True,
        'tools.response_headers.headers': [('Content-Type', 'application/json')],
        'tools.encode.on': True,
        'tools.encode.encoding': 'utf-8'
    }
}
cherrypy.quickstart(LiSEHandleWebService(args.world, args.code), '/', conf)
