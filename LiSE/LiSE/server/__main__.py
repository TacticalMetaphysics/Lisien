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
