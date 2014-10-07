# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
import LiSE
import sys

import argparse


parser = argparse.ArgumentParser(
    description='Pick a database and language'
)
parser.add_argument('-w', '--world')
parser.add_argument('-c', '--code')
parser.add_argument('-l', '--language')
parser.add_argument('maindotpy')


def lise():
    print(sys.argv)

    parsed = parser.parse_args(sys.argv)

    print("Starting ELiDE with world {}, code {}, path {}".format(
        parsed.world, parsed.code, LiSE.__path__[-1]))

    cli_args = {'LiSE': {}, 'ELiDE': {}}
    for arg in 'world', 'code', 'language':
        if getattr(parsed, arg):
            cli_args[arg]['LiSE'] = getattr(parsed, arg)
    argv = list(sys.argv)
    for o in '-w', '--world', '-c', '--code', '-l', '--language':
        try:
            i = argv.index(o)
            del argv[i:i+1]
        except ValueError:
            pass
    sys.argv = argv
    from ELiDE.app import ELiDEApp
    app = ELiDEApp(
        cli_args=cli_args
    )
    app.run()

if __name__ == '__main__':
    lise()
