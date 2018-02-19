.. allegedb documentation master file, created by
   sphinx-quickstart on Mon Feb 19 10:26:25 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to allegedb's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Object relational mapper for graphs with in-built revision control.

allegedb serves its own special variants on the networkx graph classes:
Graph, DiGraph, MultiGraph, and MultiDiGraph. Every change to them is
stored in an SQL database.

This means you can keep multiple versions of one set of graphs and
switch between them without the need to save, load, or run git-checkout.
Just point the ORM at the correct branch and revision, and all of the
graphs in the program will change. All the different branches and
revisions remain in the database to be brought back when needed.

usage
-----

::

    >>> from allegedb import ORM
    >>> orm = ORM('sqlite:///test.db')
    >>> orm.initdb()  # only necessary the first time you use a particular database
    >>> g = orm.new_graph('test')  # also new_digraph, new_multigraph, new_multidigraph
    >>> g.add_nodes_from(['spam', 'eggs', 'ham'])
    >>> g.add_edge('spam', 'eggs')
    >>> g.edge  # strings become unicode because that's the way sqlite3 rolls
    {u'eggs': {u'ham': {}, u'spam': {}}, u'ham': {u'eggs': {}}, u'spam': {u'eggs': {}}}
    >>> del g
    >>> orm.close()
    >>> del orm
    >>> orm = ORM('sqlite:///test.db')
    >>> g = orm.get_graph('test')  # returns whatever graph type you stored by that name
    >>> g.edge
    {u'eggs': {u'ham': {}, u'spam': {}}, u'ham': {u'eggs': {}}, u'spam': {u'eggs': {}}}
    >>> import networkx as nx
    >>> red = nx.random_lobster(10,0.9,0.9)
    >>> blue = orm.new_graph('red', red)  # initialize with data from the given graph
    >>> red.edge == blue.edge
    True
    >>> orm.rev = 1
    >>> blue.add_edge(17, 15)
    >>> red.edge = blue.edge
    False
    >>> orm.rev = 0  # undoing what I did when rev-1
    >>> red.edge == blue.edge
    True
    >>> orm.rev = 0
    >>> orm.branch = 'test'    # navigating to a branch for the first time creates that branch
    >>> orm.rev = 1
    >>> red.edge == blue.edge
    True
    >>> orm.branch = 'trunk'
    >>> red.edge == blue.edge
    False

allegedb
---------
.. automodule:: allegedb
   :members:

cache
-----
.. automodule:: allegedb.cache
   :members:

graph
-----
.. automodule:: allegedb.graph
   :members:

query
-----
.. automodule:: allegedb.query
   :members:

wrap
----
.. automodule:: allegedb.wrap
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
