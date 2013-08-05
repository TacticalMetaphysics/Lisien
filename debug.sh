# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013 Zachary Spector,  zacharyspector@gmail.com
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LD_LIBRARY_PATH="$dir/lib" PYTHONPATH="$dir/python-igraph*/" pdb lise.py
