.. allegedb documentation master file, created by
   sphinx-quickstart on Mon Feb 19 10:26:25 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

allegedb
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

State container and object-relational mapper for versioned graphs.

allegedb serves its own special variant on the networkx
DiGraph class (with more to come). Every change to them is
stored in an SQL database.

This means you can keep multiple versions of one set of graphs and
switch between them without the need to save, load, or run git-checkout.
Just point the ORM at the correct branch and turn, and all of the
graphs in the program will change. All the different branches and
revisions remain in the database to be brought back when needed.

usage
-----

::

    >>> from LiSE.allegedb import ORM
    >>> orm = ORM('sqlite:///test.db')
    >>> g = orm.new_digraph('test')
    >>> g.add_nodes_from(['spam', 'eggs', 'ham'])
    >>> g.add_edge('spam', 'eggs')
    >>> g.adj
    <LiSE.allegedb.graph.DiGraphSuccessorsMapping object containing {'ham': {}, 'eggs': {}, 'spam': {'eggs': {}}}>
    >>> del g
    >>> orm.close()
    >>> del orm
    >>> orm = ORM('sqlite:///test.db')
    >>> g = orm.graph['test']
    >>> g.adj
    <LiSE.allegedb.graph.DiGraphSuccessorsMapping object containing {'ham': {}, 'eggs': {}, 'spam': {'eggs': {}}}>
    >>> import networkx as nx
    >>> red = nx.random_lobster(10, 0.9, 0.9)
    >>> blue = orm.new_digraph('blue', red)  # initialize with data from the given graph
    >>> red.adj == blue.adj
    True
    >>> orm.turn = 1
    >>> blue.add_edge(17, 15)
    >>> red.adj == blue.adj
    False
    >>> orm.turn = 0  # undoing what I did when turn=1
    >>> red.adj == blue.adj
    True
    >>> orm.branch = 'test'    # navigating to a branch for the first time creates that branch
    >>> orm.turn = 1
    >>> red.adj == blue.adj
    True
    >>> orm.branch = 'trunk'
    >>> red.adj == blue.adj
    False

ORM
---
.. automodule:: LiSE.allegedb
   :members:

cache
-----
.. automodule:: LiSE.allegedb.cache
   :members:

graph
-----
.. automodule:: LiSE.allegedb.graph
   :members:

query
-----
.. automodule:: LiSE.allegedb.query
   :members:

wrap
----
.. automodule:: LiSE.allegedb.wrap
   :members:

