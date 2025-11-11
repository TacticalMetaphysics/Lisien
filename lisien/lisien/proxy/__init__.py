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
"""Proxy objects to access lisien entities from another process.

Each proxy class is meant to emulate the equivalent lisien class,
and any change you make to a proxy will be made in the corresponding
entity in the lisien core.

To use these, first instantiate an ``EngineProcessManager``, then
call its ``start`` method with the same arguments you'd give a real
``Engine``. You'll get an ``EngineProxy``, which acts like the underlying
``Engine`` for most purposes.

``EngineHandle`` is a fairly thin wrapper around a real Lisien ``Engine``
that exposes all functionality that proxies need in simple methods.

"""

from .handle import EngineHandle as EngineHandle
from .manager import EngineProxyManager as EngineProxyManager
