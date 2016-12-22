from multiprocessing import freeze_support
import sys
import os
wd = os.getcwd()
sys.path.extend([wd + '/LiSE', wd + '/ELiDE', wd + '/../gorm'])
from ELiDE.app import ELiDEApp
import kivy._event

if __name__ == '__main__':
    freeze_support()

    def on_engine(inst, val):
        val.handle('install_module', module='examples.college')

    def get_application_config(*args):
        return wd + '/ELiDE.ini'


    app = ELiDEApp()
    app.get_application_config = get_application_config
    app.bind(engine=on_engine)
    app.run()
