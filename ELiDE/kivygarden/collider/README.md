garden.collider
===============

See http://kivy-garden.github.io/garden.collider/index.html for html docs.

The collider module contains classes which can be used to test membership
of a point in some space. See individual class documentation for details.

For example, using the Collide2DPoly class we can test whether points fall
within a general polygon, e.g. a simple triangle::

    >>> collider = Collide2DPoly([10., 10., 20., 30., 30., 10.],\
                                 cache=True)
    >>> (0.0, 0.0) in collider
    False
    >>> (20.0, 20.0) in collider
    True
