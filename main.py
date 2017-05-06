from multiprocessing import freeze_support
from ELiDE.app import ELiDEApp
import sys
import os
wd = os.getcwd()
sys.path.extend([wd + '/LiSE', wd + '/ELiDE', wd + '/allegedb'])


def get_application_config(*args):
    return wd + '/ELiDE.ini'


if __name__ == '__main__':
    freeze_support()

    app = ELiDEApp()
    app.get_application_config = get_application_config
    app.run()
