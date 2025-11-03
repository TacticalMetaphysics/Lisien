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
import os
import sys

from lisien.proxy.manager import EngineProxyManager

if __name__ == "__main__":
	if os.path.exists(sys.argv[-1]) and os.path.isfile(sys.argv[-1]):
		mgr = EngineProxyManager()
		eng = mgr.start(replay_file=sys.argv[-1], loglevel="debug")
		mgr.shutdown()
