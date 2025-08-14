# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
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
from kivy.logger import ConsoleHandler, KivyFormatter, Logger
from kivy.resources import resource_add_path

formatter = KivyFormatter("%(asctime)s [%(levelname)-7s] %(message)s")
for handler in Logger.handlers:
	if not isinstance(handler, ConsoleHandler):
		handler.setFormatter(formatter)

resource_add_path(__path__[0])
resource_add_path(__path__[0] + "/assets")
resource_add_path(__path__[0] + "/assets/rltiles")
resource_add_path(__path__[0] + "/assets/kenney1bit")

__all__ = [
	"graph",
	"grid",
	"app",
	"card",
	"dialog",
	"game",
	"menu",
	"spritebuilder",
	"calendar",
]
