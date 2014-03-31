# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com


class Implicator(object):
    """An event handler for those events that take place in the simulated world.

    Whenever new ticks are added to a timeline, the Implicator will take a look at the world-state during each of them in turn. Boolean functions called 'causes' will be called to evaluate whether their associated Event should """
    
