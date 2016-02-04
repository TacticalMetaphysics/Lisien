# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Initialize the kobold test and launch the GUI."""
from ELiDE.app import ELiDEApp
from kobold import clear_off, mkengine, inittest

clear_off()
with mkengine() as engine:
    inittest(engine)


app = ELiDEApp()
app.run()
