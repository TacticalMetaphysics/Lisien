============
AST Unparser
============

.. image:: https://badge.fury.io/py/astunparse.png
    :target: http://badge.fury.io/py/astunparse

.. image:: https://travis-ci.org/simonpercivall/astunparse.png?branch=master
    :target: https://travis-ci.org/simonpercivall/astunparse

.. image:: https://readthedocs.org/projects/astunparse/badge/
    :target: https://astunparse.readthedocs.org/

An AST unparser for Python.

This is a factored out version of ``unparse`` found in the Python
source distribution; under Demo/parser in Python 2 and under Tools/parser
in Python 3.

Basic example::

    import inspect
    import ast
    import astunparse

    # get back the source code
    astunparse.unparse(ast.parse(inspect.getsource(ast)))

    # get a pretty-printed dump of the AST
    astunparse.dump(ast.parse(inspect.getsource(ast)))


This library is single-source compatible with Python 2.6 through Python 3.5. It
is authored by the Python core developers; I have simply merged the Python 2.7
and the Python 3.5 source and test suites, and added a wrapper. This factoring
out is to provide a library implementation that supports both versions.

Added to this is a pretty-printing ``dump`` utility function.

The test suite both runs specific tests and also roundtrips much of the
standard library.

Extensions and Alternatives
---------------------------

Similar projects include:

    * codegen_
    * astor_
    * astmonkey_
    * astprint_

None of these roundtrip much of the standard library and fail several of the basic
tests in the ``test_unparse`` test suite.

This library uses mature and core maintained code instead of trying to patch
existing libraries. The ``unparse`` and the ``test_unparse`` modules
are under the PSF license.

Extensions include:

    * typed-astunparse: extends astunparse to support type annotations.

* Documentation: http://astunparse.rtfd.org.

Features
--------

* unparses Python AST.
* pretty-prints AST.


.. _codegen: https://github.com/andreif/codegen
.. _astor: https://github.com/berkerpeksag/astor
.. _astmonkey: https://github.com/konradhalas/astmonkey
.. _astprint: https://github.com/Manticore/astprint


Changelog
=========

Here's the recent changes to AST Unparser.

1.5.0 - 2017-02-05
~~~~~~~~~~~~~~~~~~

* Python 3.6 compatibility
* bugfix: correct argparser option type

1.4.0 - 2016-06-24
~~~~~~~~~~~~~~~~~~

* Support for the ``async`` keyword
* Support for unparsing "Interactive" and "Expression" nodes

1.3.0 - 2016-01-17
~~~~~~~~~~~~~~~~~~

* Python 3.5 compatibility

1.2.0 - 2014-04-03
~~~~~~~~~~~~~~~~~~

* Python 2.6 through 3.4 compatibility
* A new function ``dump`` is added to return a pretty-printed version
  of the AST. It's also available when running ``python -m astunparse``
  as the ``--dump`` argument.

1.1.0 - 2014-04-01
~~~~~~~~~~~~~~~~~~

* ``unparse`` will return the source code for an AST. It is pretty
  feature-complete, and round-trips the stdlib, and is compatible with
  Python 2.7 and Python 3.4.

  Running ``python -m astunparse`` will print the round-tripped source
  for any python files given as argument.


