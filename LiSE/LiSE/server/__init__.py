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
import threading
import logging
from queue import Queue
from ..handle import EngineHandle


class LiSEHandleWebService(object):
    exposed = True

    def __init__(self, *args, **kwargs):
        if 'logger' in kwargs:
            self.logger = kwargs['logger']
        else:
            self.logger = kwargs['logger'] = logging.getLogger(__name__)
        self.cmdq = kwargs['cmdq'] = Queue()
        self.outq = kwargs['outq'] = Queue()
        self._handle_thread = threading.Thread(
            target=self._run_handle_forever, args=args, kwargs=kwargs,
            daemon=True
        )
        self._handle_thread.start()

    @staticmethod
    def _run_handle_forever(*args, **kwargs):
        cmdq = kwargs.pop('cmdq')
        outq = kwargs.pop('outq')
        logger = kwargs.pop('logger')
        setup = kwargs.pop('setup', None)
        logq = Queue()

        def log(typ, data):
            if typ == 'command':
                (cmd, args) = data
                logger.debug(
                    "LiSE thread {}: calling {}{}".format(
                        threading.get_ident(),
                        cmd,
                        tuple(args)
                    )
                )
            else:
                logger.debug(
                    "LiSE thread {}: returning {} (of type {})".format(
                        threading.get_ident(),
                        data,
                        repr(type(data))
                    )
                )

        def get_log_forever(logq):
            (level, data) = logq.get()
            getattr(logger, level)(data)

        engine_handle = EngineHandle(args, kwargs, logq)
        if setup:
            setup(engine_handle._real)
        handle_log_thread = threading.Thread(
            target=get_log_forever, args=(logq,), daemon=True
        )
        handle_log_thread.start()
        while True:
            inst = cmdq.get()
            if inst == 'shutdown':
                handle_log_thread.join()
                cmdq.close()
                outq.close()
                return 0
            cmd = inst.pop('command')
            silent = inst.pop('silent', False)
            log('command', (cmd, args))
            response = getattr(engine_handle, cmd)(**inst)
            if silent:
                continue
            log('result', response)
            outq.put(engine_handle._real.listify(response))

    @cherrypy.tools.accept(media='application/json')
    @cherrypy.tools.json_out()
    def GET(self):
        return cherrypy.session['LiSE_response']

    @cherrypy.tools.json_out()
    def POST(self, **kwargs):
        silent = kwargs.get('silent', False)
        self.cmdq.put(kwargs)
        if silent:
            return None
        response = self.outq.get()
        cherrypy.session['LiSE_response'] = response
        return response

    def PUT(self, silent=False, **kwargs):
        silent = silent
        self.cmdq.put(kwargs)
        if not silent:
            cherrypy.session['LiSE_response'] = self.outq.get()

    def DELETE(self):
        cherrypy.session.pop('LiSE_response', None)
