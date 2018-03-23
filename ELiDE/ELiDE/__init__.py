# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com

from kivy.resources import resource_add_path
resource_add_path(__path__[0] + '/assets')

__all__ = ['board', 'app', 'card', 'dialog', 'game', 'spritebuilder']