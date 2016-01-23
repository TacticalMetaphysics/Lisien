from multiprocessing import freeze_support
from ELiDE.__main__ import elide
import kivy._event

if __name__ == '__main__':
    freeze_support()
    elide()
