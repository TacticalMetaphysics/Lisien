from ELiDE.app import ELiDEApp
from kobold import clear_off, mkengine, inittest

clear_off()
with mkengine() as engine:
    inittest(engine)


app = ELiDEApp()
app.run()
