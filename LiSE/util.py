# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from array import array
import struct
from collections import (
    MutableMapping,
    OrderedDict)
from math import sqrt, hypot, atan, pi, sin, cos
from operator import itemgetter
from re import match, compile, findall
from sqlite3 import IntegrityError


"""Common utility functions and data structures.

The most important are Skeleton, a mapping used to store and maintain
all game data; and SaveableMetaclass, which generates
SQL from metadata declared as class atttributes.

"""

### Constants

int2pytype = {
    0: bool,
    1: int,
    2: float,
    3: unicode}
"""Arbitrary numerical codes for inbuilt Python types"""

pytype2int = dict([
    (value, key) for (key, value) in int2pytype.iteritems()
])

packed_str_len = 128
"""When packing a string field of a bone into an array, how long
should the string be made? It will be padded or truncated as
needed."""

phi = (1.0 + sqrt(5))/2.0
"""The golden ratio."""

portex = compile("Portal\((.+?)->(.+?)\)")
"""Regular expression to recognize portals by name"""

### End constants
### Begin metadata

schemata = {}
"""Map the name of each table to its schema."""

colnames = {}
"""Map the name of each table to the names of its fields, in a tuple."""

colnamestr = {}
"""Map the name of each table to the names of its fields, in a string."""

primarykeys = {}
"""Map the name of each table to the names of the fields in its
primary key, in a tuple."""

tabclas = {}
"""Map the name of each table to the class it was declared in."""

saveables = []
"""Tuples of information about saveable classes. These may be used to
apply the database schema."""

saveable_classes = []
"""Classes that use SaveableMetaclass"""

### End metadata
### Begin functions


def upbranch(closet, bones, branch, tick):
    started = False
    first = None
    for bone in bones:
        if bone.tick >= tick:
            started = True
            yield bone._replace(branch=branch)
        if not started:
            assert(bone.tick < tick)
            first = bone
    if first is not None:
        yield first._replace(
            branch=branch, tick=tick)


def selectif(skel, key):
    if key is None:
        for sk in skel.itervalues():
            yield sk
    else:
        try:
            yield skel[key]
        except (KeyError, IndexError):
            return

### End functions


class BoneMetaclass(type):
    """Metaclass for the creation of :class:`Bone` and its subclasses."""

    def __new__(metaclass, clas, parents, attrs):
        """Create a new Bone class, based on the field declarations in the
        attribute _field_decls.

        _field_decls is a list of triples. In the triples, the first
        value is the field name, the second is its type (a type
        object), and the third is the default value.

        Regardless of the type of a field, all fields may be set to
        None. This is to make Bones usable in queries, which would be
        pretty pointless if you could not leave the data unspecified.

        """
        def __new__(_cls, *args, **kwargs):
            """Create new instance of {}""".format(clas)
            if len(args) == len(_cls._fields):
                return tuple.__new__(_cls, args)
            elif len(args) > 0:
                raise ValueError("Wrong number of values")
            values = []
            for fieldn in _cls._fields:
                if fieldn in kwargs:
                    if type(kwargs[fieldn]) not in (
                            _cls._types[fieldn], type(None)):
                        kwargs[fieldn] = _cls._types[fieldn](kwargs[fieldn])
                    values.append(kwargs[fieldn])
                else:
                    values.append(_cls._defaults[fieldn])
            r = tuple.__new__(_cls, tuple(values))
            return r

        @classmethod
        def getfmt(cls):
            return cls.structtyp.format

        @classmethod
        def getbytelen(cls):
            return cls.structtyp.size

        @classmethod
        def _make(cls, iterable):
            """Make a new {} object from a sequence or iterable""".format(
                clas)
            return tuple.__new__(cls, iterable)

        @property
        def packed(self):
            """Return a string of data packed according to self.format."""
            args = []
            for datum in self:
                if isinstance(datum, unicode):
                    args.append(str(datum))
                else:
                    args.append(datum)
            return self.structtyp.pack(*args)

        def _pack_into(self, arr, i):
            """Put myself into logical position i of array arr.

            Actual position is determined by multiplying i by my
            record length.

            """
            size = self.structtyp.size
            pos = i * size
            args = [arr, pos]
            for field in self._fields:
                datum = getattr(self, field)
                if isinstance(datum, unicode):
                    args.append(str(datum))
                elif datum is None:
                    if self._defaults[field] is None:
                        if self._types[field] in (str, unicode):
                            args.append('\x00' * packed_str_len)
                        else:
                            args.append(0)
                    else:
                        args.append(self._defaults[field])
                else:
                    args.append(datum)
            self.structtyp.pack_into(*args)

        def denull(self):
            """Return a copy of myself with the nulls stripped out of
            the strings.

            """
            denulled = {}
            for field in self._fields:
                if isinstance(getattr(self, field), str):
                    denulled[field] = unicode(
                        getattr(self, field)).strip('\x00')
                    if denulled[field] == '':
                        denulled[field] = None
            return self._replace(**denulled)

        @classmethod
        def _null(cls):
            """Return an instance of {} with all fields set to
            None.""".format(cls.__name__)
            aargh = [None] * len(cls._fields)
            return cls(*aargh)

        @classmethod
        def _unpack(cls, data):
            """Return a new instance of {} with data from
            the buffer given.""".format(cls)
            return cls(*cls.structtyp.unpack(data)).denull()

        @classmethod
        def _unpack_from(cls, i, arr):
            """Return a new instance of {} using the data at the logical
            position ``i`` in array ``arr``.

            i will be multiplied by my record length before unpacking.

            """
            bytelen = cls.structtyp.size
            pos = i * bytelen
            data = cls.structtyp.unpack_from(arr, pos)
            return cls(*data).denull()

        def __repr__(self):
            """Return a nicely formatted representation string"""
            return "{}({})".format(clas, ", ".join(
                ["{}={}".format(field, getattr(self, field))
                 for field in self._fields]))

        def _asdict(self):
            """Return a new OrderedDict which maps field names
            to their values"""
            return OrderedDict(zip(self._fields, self))

        def _replace(self, **kwds):
            """Return a new {} object replacing specified fields with new
            values""".format(clas)
            result = self._make(map(kwds.pop, self._fields, self))
            if kwds:
                raise ValueError("Got unexpected field names: {}".format(
                    kwds.keys()))
            return result

        def __getnewargs__(self):
            """Return self as a plain tuple. Used by copy and pickle."""
            return tuple(self)

        atts = {"__new__": __new__,
                "_make": _make,
                "__repr__": __repr__,
                "_asdict": _asdict,
                "_replace": _replace,
                "__getnewargs__": __getnewargs__,
                "__dict__": property(_asdict),
                "_fields": [],
                "_types": {},
                "_defaults": {},
                "_name": clas,
                "_null": _null,
                "getfmt": getfmt,
                "getbytelen": getbytelen,
                "packed": packed,
                "_unpack": _unpack,
                "_unpack_from": _unpack_from,
                "_pack_into": _pack_into,
                "denull": denull}
        atts.update(attrs)
        if '_no_fmt' not in atts:
            fmt = bytearray('@')
            for (field_name, field_type, default) in atts["_field_decls"]:
                if field_type in (unicode, str):
                    fmt.extend('{}s'.format(packed_str_len))
                elif field_type is int:
                    fmt.append('l')
                elif field_type is float:
                    fmt.append('d')
                elif field_type is bool:
                    fmt.append('b')
                else:
                    raise TypeError(
                        "Trying to make a format string; "
                        "don't understand the type "
                        "{}".format(field_type))
            atts["structtyp"] = struct.Struct(str(fmt))
            atts["format"] = atts["structtyp"].format
            atts["size"] = atts["structtyp"].size

        i = 0
        for (field_name, field_type, default) in atts["_field_decls"]:
            atts["_fields"].append(field_name)
            atts[field_name] = property(
                itemgetter(i),
                doc="Alias for field number {}".format(i))
            atts["_types"][field_name] = field_type
            atts["_defaults"][field_name] = default
            i += 1
        return type.__new__(metaclass, clas, parents, atts)


class Bone(tuple):
    """A named tuple with an odd interface, which can be packed into
    an array.

    :class:`Bone` is meant to represent records in a database,
    including all the restrictions that come with the database
    schema. It is used to cache these records, speeding up access.

    Each subclass of :class:`Bone` has its own fields, each with a
    particular type. These are declared in the class attribute
    ``_field_decls``, a list of triples, each containing a field name,
    its type, and its default value.

    Instances of :class:`Bone` may be constructed with keyword
    arguments. All, some, or none of the fields may be filled this
    way; those whose names are not used as keywords will be set to
    their default value. Setting a keyword argument to ``None``,
    explicitly, will always result in ``None`` occupying that field.

    To pack a :class:`Bone` into an array, use
    :method:`_pack_into`. This calls therelevant :module:`struct`
    method, supplying a format string appropriate to the field types
    specified at class creation. Retrieve the :class:`Bone` later
    using :method:`unpack_from`.

    :class:`Bone` should not be instantiated directly. Subclass it
    instead. For convenience, the class method :method:`subclass` will
    return an appropriate subclass. :method:`structless_subclass` will
    do the same, but the subclass will not be packable into arrays.

    """
    __metaclass__ = BoneMetaclass
    _field_decls = []

    def __init__(self, *args, **kwargs):
        """Refuse to initialize :class:`Bone` directly.

        Please use the class method :method:`subclass` (or
        :method:`structless_subclass` for weird data) to make your
        own.

        """
        if self.__class__ is Bone:
            raise NotImplementedError("Bone is an abstract class.")
        super(Bone, self).__init__(*args, **kwargs)

    @classmethod
    def subclass(cls, name, decls):
        """Return a subclass of :class:`Bone` named ``name`` with field
        declarations ``decls``.

        Field declarations look like:

        ``(field_name, field_type, default)``

        field_name is a string; it will be the name of a property of the class.

        field_type is a type object. It should be one of the types
        supported by the :module:`struct` module, or you will get
        errors. You can still use ``structless_subclass`` if you need
        weird types; you just won't be able to keep your bones in
        arrays.

        ``default`` is the default value of the field, for when you
        construct a bone with keyword arguments that do not include
        this field.

        """
        d = {"_field_decls": decls}
        return type(name, (cls,), d)

    @classmethod
    def structless_subclass(cls, name, decls):
        """Return a subclass of :class:`Bone` named ``name`` with field
        declarations ``decls``. This subclass will not be packable
        into arrays.

        Field declarations look like:

        ``(field_name, field_type, default)``

        field_name is a string; it will be the name of a property of the class.

        field_type is a type object. The values of this field must
        match it, except that any field may be ``None``.

        ``default`` is the default value of the field, for when you
        construct a bone with keyword arguments that do not include
        this field.

        """
        d = {"_field_decls": decls,
             "_no_fmt": True}
        return type(name, (cls,), d)


class PlaceBone(Bone):
    _field_decls = [("host", unicode, None), ("place", unicode, None),
                    ("branch", int, 0), ("tick", int, 0)]


class Skeleton(MutableMapping):
    """A tree structure whose leaves correspond directly to individual
    database records.

    Skeleton is used to store a cache of some or all of the LiSE
    database. It does not, itself, synchronize with the
    database--Closet and SaveableMetaclass handle that--but it
    supports union and difference operations, as though it were a
    set. This makes it easy to decide which records to save.

    Nearly everything in LiSE keeps its data in a single, enormous
    Skeleton, which itself is kept in an instance of Closet. Whenever
    you want a LiSE object, you should get it by calling some
    appropriate method in the single instance of Closet used by that
    instance of LiSE. The method will require keys for
    arguments--these are the same keys you would use to look up a
    record in one of the tables that the object refers to. The object
    you get this way will not really contain any data apart from the
    key and the Closet--whenever it needs to know something, it will
    ask Closet about it, and Closet will look up the answer in
    Skeleton, first loading it in from disc if necessary.

    Apart from simplifying the process of saving, this approach also
    makes LiSE's time travel work. The keys that identify LiSE objects
    are only partial keys--they do not specify time. Every record in
    the database and the Skeleton is also keyed by the time at which
    that record was written--not "real" time, but simulated time,
    measured in ticks that represent the smallest significant
    time-span in your simulation. They are also keyed with a branch
    number, indicating which of several alternate histories the record
    is about. When you refer to properties of a LiSE object, you get
    data from only one branch of one tick--the most recent data with
    respect to the time the simulation is at, in the branch the
    simulation is at, but ignoring any data whose tick is later than
    the tick the simulation is at. The simulation's "present" branch
    and tick are attributes of its Closet. The user may set them
    arbitrarily, and thereby travel through time.

    The "leaves" at the lowest level of a Skeleton are instances of
    some subclass of :class:`Bone`. The API for :class:`Bone` is made
    to resemble an ordinary Python object with properties full of
    data, with type checking. Treat it that way; the only caveat is
    that any :class:`Skeleton` with any type of :class:`Bone` in its
    values cannot hold any other type. This helps with data integrity,
    as the corresponding table in the database has that
    :class:`Bone`'s fields and none other. It also allows for
    optimization using :module:`array` and :module:`struct`. See the
    documentation of :class:`Bone` for details.

    Iteration over a :class:`Skeleton` is a bit unusual. The usual
    iterators over mappings are available, but they will proceed in
    ascending order of the keys if, and only if, the keys are
    integers. Otherwise they are in the same order as a
    dictionary. :class:`Skeleton` also provides the special generator
    method :method:`iterbones`, which performs a depth-first
    traversal, yielding only the :class:`Bone`s.

    Several methods are made purely for the convenience of time
    travel, particularly :method:`value_during`, which assumes that
    the keys are ticks, and returns the :class:`Bone` of the latest
    tick less than or equal to the present tick.

    There is a primitive event handling infrastructure built into
    :class:`Skeleton`. Functions in the list :attribute:`listeners`
    will be called with the :class:`Skeleton` object, the key, and the
    value of each assignment to the :class:`Skeleton`, or to any other
    :class:`Skeleton` contained by that one, however
    indirectly. Likewise with each deletion, though in that case, the
    value is not supplied. This feature exists for the convenience of
    the user interface code.

    """
    def __init__(self, content=None, name="", parent=None):
        """Technically all of the arguments are optional but you should really
        specify them whenever it's reasonably sensible to do
        so. ``content`` is a mapping to crib from, ``parent`` is who to
        propagate events to, and ``name`` is mostly for printing.

        """
        self._set_listeners = []
        self._del_listeners = []
        """Functions to call when something changes, either in my content, or
        in that of some :class:`Skeleton` I contain, however indirectly.

        """
        self.name = name
        """A string for representing myself."""
        self.parent = parent
        """Some other :class:`Skeleton` to propagate events to. Can be
        ``None``.

        """
        self.content = {}
        """My data, a :type:`dict` by default. It may become a :type:`list` or
        :type:`array` when its first key is assigned."""
        if content is not None:
            self._populate_content(content)

    def _populate_content(self, content):
        """Fill myself with content."""
        if content in (None, {}, []):
            return
        if content.__class__ in (dict, self.__class__):
            kitr = content.iteritems()
        else:
            kitr = ListItemIterator(content)
        for (k, v) in kitr:
            if hasattr(v, 'content'):
                # It's a Skeleton.
                # Add a copy of it, unless there's no data to copy,
                # in which case don't bother.
                if len(v.content) == 0:
                    self[k] = None
                else:
                    self[k] = v.copy()
                    self[k].name = k
                    self[k].parent = self
            elif v is None or issubclass(v.__class__, Bone):
                self[k] = v
            else:
                self[k] = self.__class__(
                    content=v, name=k, parent=self)

    def __contains__(self, k):
        """Check if I have the given key"""
        if isinstance(self.content, dict):
            return k in self.content
        else:
            return k in self.ikeys

    def __getitem__(self, k):
        """Get item from ``self.content``, which may be :type:`dict`,
        :type:`list`, or :type:`array`. Unpack it in the latter
        case.

        """
        if isinstance(self.content, list):
            return self.content[k]
        elif isinstance(self.content, dict):
            return self.content[k]
        else:
            # I'm full of Bones
            return self.bonetype._unpack_from(k, self.content)

    def __setitem__(self, k, v):
        """Set item in ``self.content``, which may be :type:`dict`,
        :type:`list`, or :type:`array`. Pack it in the latter case.

        """
        if len(self.content) <= 1:
            # It's permissible to assign a Bone to a Skeleton with
            # content already in, provided the Bone immediately
            # overwrites it.
            if issubclass(v.__class__, Bone):
                self.bonetype = type(v)
                """This is the subclass of :class:`Bone` that I contain.
                Once set, only this ``bonetype`` may occupy
                ``self.content``."""
            if isinstance(k, int):
                if not hasattr(self, 'ikeys'):
                    self.ikeys = []
                    """A set of indices in ``self.content`` that contain
                    legitimate data. They will be treated like keys. Other
                    indices are regarded as empty.

                    """
                if hasattr(self, 'bonetype'):
                    if len(self.content) > 0:
                        assert(len(self.content) == 1)
                        if isinstance(self.content, dict):
                            old = next(self.content.itervalues())
                        else:
                            old = self.content[0]
                        self.content = array('c')
                        if isinstance(old, self.bonetype):
                            self.content.extend('\x00' * old.size)
                            old._pack_into(self.content, 0)
                elif len(self.content) > 0:
                    assert(len(self.content) == 1)
                    if isinstance(self.content, dict):
                        assert(next(self.content.iterkeys()) == 0)
                        old = next(self.content.itervalues())
                    else:
                        old = self.content[0]
                    self.content = [old]
                else:
                    self.content = []
        if hasattr(self, 'ikeys') and v is not None:
            if len(self.ikeys) == 0:
                self.ikeys = [k]
            else:
                i = 0
                end = True
                for ik in self.ikeys:
                    if ik > k:
                        self.ikeys.insert(i, k)
                        end = False
                        break
                    i += 1
                if end:
                    self.ikeys.append(k)
        if hasattr(self, 'bonetype'):
            if not isinstance(v, self.bonetype):
                raise TypeError(
                    "Skeletons that contain one Bone type may "
                    "only contain that type.")
            if isinstance(self.content, array):
                # Pad self.content with nulls until it's big enough.
                diff = (k+1) * v.size - len(self.content)
                if diff > 0:
                    self.content.extend('\x00' * diff)
        elif isinstance(self.content, list):
            # Pad self.content with None until it's big enough.
            diff = k + 1 - len(self.content)
            if diff > 0:
                self.content.extend([None] * diff)
        ### actually set content to value
        if issubclass(v.__class__, Bone):
            if isinstance(self.content, array):
                v._pack_into(self.content, k)
            else:
                self.content[k] = v
        elif isinstance(v, self.__class__):
            v.name = k
            v.parent = self
            self.content[k] = v
        else:
            self.content[k] = self.__class__(v, name=k, parent=self)
        for listener in self._set_listeners:
            listener(self.parent, self, k, v)
        if hasattr(self.parent, 'on_child_set'):
            self.parent.on_child_set(self, k, v)

    def on_child_set(self, child, k, v):
        """Call all my listeners with args (child, k, v)."""
        for listener in self._set_listeners:
            listener(self, child, k, v)
        if hasattr(self.parent, 'on_child_set'):
            self.parent.on_child_set(child, k, v)

    def register_set_listener(self, fun):
        """Register a function to be called when a value is set on this or a
        child."""
        self._set_listeners.append(fun)

    def __delitem__(self, k):
        """If ``self.content`` is a :type:`dict`, delete the key in the usual
        way. Otherwise, remove the key from ``self.ikeys``."""
        if isinstance(self.content, dict):
            del self.content[k]
        else:
            self.ikeys.remove(k)
        for listener in self._del_listeners:
            listener(self.parent, self, k)
        if hasattr(self.parent, 'on_child_del'):
            self.parent.on_child_del(self, k)

    def on_child_del(self, child, k):
        """Call all my listeners with args (child, k)."""
        for listener in self._del_listeners:
            listener(self, child, k)
        if hasattr(self.parent, 'on_child_del'):
            self.parent.on_child_del(child, k)

    def register_del_listener(self, fun):
        """Register a function to be called when an element is deleted in this
        or a child."""
        self._del_listeners.append(fun)

    def __iter__(self):
        """Iterate over my keys--which, if ``self.content`` is not a
        :type:`dict`, should be taken from ``self.ikeys`` and sorted first."""
        if isinstance(self.content, dict):
            return iter(self.content)
        else:
            return iter(self.ikeys)

    def __len__(self):
        """Return the number of "live" data that I have. If ``self.content``
        is not a :type:`dict`, that means the number of keys that
        point to something meaningful.

        """
        if isinstance(self.content, dict):
            return len(self.content)
        else:
            return len(self.ikeys)

    def __repr__(self):
        """If ``self.content`` is an :type:`array`, unpack the lot of
        it for show. Otherwise just return ``repr(self.content)``."""
        if isinstance(self.content, array):
            return repr([bone for bone in self.itervalues()])
        return repr(self.content)

    def __iadd__(self, other):
        """Wrapper for ``self.update`` that returns ``self``."""
        self.update(other)
        return self

    def __add__(self, other):
        """Return a copy of ``self`` that's been updated with ``other``."""
        selfie = self.copy()
        selfie.update(other)
        return selfie

    def __isub__(self, other):
        """Remove everything in me and my children that is in ``other``
        or its children. Return myself."""
        for (k, v) in other.iteritems():
            if k not in self:
                continue
            elif v == self[k]:
                del self[k]
            elif isinstance(self[k], Skeleton) and isinstance(v, Skeleton):
                self[k] -= v
            # otherwise, just keep it
        return self

    def __sub__(self, other):
        """Return a copy of myself that's had anything matching anything in
        ``other`` removed."""
        selfie = self.copy()
        selfie -= other
        return selfie

    def __str__(self):
        """Return my name, stringly."""
        return str(self.name)

    def __unicode__(self):
        """Return my name, unicodely."""
        return unicode(self.name)

    def keys(self):
        """If I contain a :type:`dict`, use its keys, otherwise
        ``self.ikeys``"""
        if isinstance(self.content, dict):
            return self.content.keys()
        else:
            return self.ikeys

    def key_before(self, k):
        """Return my largest key that is smaller than ``k``."""
        if hasattr(self, 'ikeys'):
            if k in self.ikeys:
                iki = self.ikeys.index(k)
                if iki == 0:
                    return None
                else:
                    return self.ikeys[iki-1]
            else:
                return max([j for j in self.ikeys if j < k])
        return max([j for j in self.content.keys() if j < k])

    def key_or_key_before(self, k):
        """Return my highest key less than or equal to ``k``."""
        if hasattr(self, 'ikeys'):
            if k in self.ikeys:
                return k
            else:
                return self.key_before(k)
        if k in self.content:
            return k
        else:
            return self.key_before(k)

    def value_during(self, k):
        """Return my value at ``k``, or the most recent before ``k`` if I
        don't have one exactly at ``k``.

        """
        try:
            return self[self.key_or_key_before(k)]
        except ValueError:
            return None

    def key_after(self, k):
        """Return my smallest key larger than ``k``."""
        if hasattr(self, 'ikeys'):
            if k in self.ikeys:
                iki = self.ikeys.index(k)
                if iki == len(self.ikeys) - 1:
                    return None
                else:
                    return self.ikeys[iki + 1]
            else:
                return min([j for j in self.ikeys if j > k])
        return min([j for j in self.content.keys() if j > k])

    def key_or_key_after(self, k):
        """Return ``k`` if it's a key I have, or else my
        smallest key larger than ``k``."""
        if hasattr(self, 'ikeys'):
            if k in self.ikeys:
                return k
            else:
                return self.key_after(k)
        if k in self.content:
            return k
        else:
            return self.key_after(k)

    def copy(self):
        """Return a shallow copy of myself. Changes to the copy won't affect
        *me* but will affect any mutable types *inside* me."""
        if isinstance(self.content, array):
            r = self.__class__()
            r.bonetype = self.bonetype
            r.name = self.name
            r.content = self.content
            r.ikeys = list(self.ikeys)
            return r
        elif hasattr(self, 'bonetype'):
            r = {}
            if isinstance(self.content, list):
                for k in self.ikeys:
                    r[k] = self.content[k]
            else:
                for (k, v) in self.content.iteritems():
                    r[k] = v
            return self.__class__(content=r)
        elif isinstance(self.content, list):
            r = {}
            for k in self.ikeys:
                if isinstance(self.content[k], self.__class__):
                    r[k] = self.content[k].copy()
                else:
                    r[k] = self.content[k]
            return self.__class__(content=r)
        else:
            r = {}
            for (k, v) in self.content.items():
                if isinstance(v, self.__class__):
                    if len(v.content) == 0:
                        r[k] = None
                    else:
                        r[k] = v.copy()
                        r[k].parent = self
                        r[k].name = k
                elif v is None or issubclass(v.__class__, Bone):
                    r[k] = v
            return self.__class__(content=r)

    def deepcopy(self):
        """Return a new :class:`Skeleton` with all of my data in it, no matter
        how many layers I have to recurse."""
        r = {}
        for (k, v) in self.iteritems():
            if hasattr(v, 'deepcopy'):
                r[k] = v.deepcopy()
            else:
                assert(issubclass(v.__class__, Bone))
                r[k] = v
        return self.__class__(
            content=r, name=self.name, parent=self.parent)

    def update(self, skellike):
        """Make my content match that of ``skellike``, which may be a
        :class:`Skeleton` or a :type:`dict`."""
        for (k, v) in skellike.iteritems():
            if issubclass(v.__class__, Bone):
                self[k] = v
            elif isinstance(v, self.__class__):
                if k in self.content:
                    self[k].update(v)
                else:
                    self[k] = v.deepcopy()
            else:
                assert(v.__class__ in (dict, list))
                if k in self.content:
                    self[k].update(v)
                else:
                    self[k] = self.__class__(
                        content=v, name=k, parent=self)

    def itervalues(self):
        if isinstance(self.content, array):
            for i in self.ikeys:
                yield self.bonetype._unpack_from(i, self.content)
        else:
            for v in super(Skeleton, self).itervalues():
                yield v

    def iterbones(self):
        """Perform a depth-first traversal over all :class:`Bone` objects I
        contain, however indirectly.

        The traversal follows the order of integer keys,
        where they are present.

        """
        if isinstance(self.content, dict):
            for contained in self.content.itervalues():
                if issubclass(contained.__class__, Bone):
                    yield contained
                else:
                    for bone in contained.iterbones():
                        yield bone
        elif hasattr(self, 'bonetype'):
            for bone in self.itervalues():
                yield bone
        else:
            for i in self.ikeys:
                for bone in self.content[i].iterbones():
                    yield bone


class SaveableMetaclass(type):
    """SQL strings and methods relevant to the tables a class is about.

    Table declarations
    ==================

    Classes with this metaclass need to be declared with an attribute
    called tables. This is a sequence of tuples. Each of the tuples is
    of length 5. Each describes a table that records what's in the
    class.

    The meaning of each tuple is thus:

    ``(name, column_declarations, primary_key, foreign_keys, checks)``

    ``name`` is the name of the table as sqlite3 will use it.

    ``column_declarations`` is a dictionary. The keys are field names, aka
    column names. Each value is the type for its field, perhaps
    including a clause like DEFAULT 0.

    ``primary_key`` is an iterable over strings that are column names as
    declared in the previous argument. Together the columns so named
    form the primary key for this table.

    ``foreign_keys`` is a dictionary. Each foreign key is a key here, and
    its value is a pair. The first element of the pair is the foreign
    table that the foreign key refers to. The second element is the
    field or fields in that table that the foreign key points to.

    ``checks`` is an iterable over strings that will end up in a CHECK(...)
    clause in sqlite3.

    A class of :class:`SaveableMetaclass` can have any number of such
    table-tuples. The tables will be declared in the order they appear
    in the tables attribute.

    Running queries
    ===============

    Classes in :class:`SaveableMetaclass` have methods to generate and
    execute SQL, but as they are not themselves database connectors,
    they require you to supply a cursor as the first argument. The
    second argument should be a dictionary (or :class:`Skeleton`) in
    which the keys are names of tables that the class knows about, and
    the values are iterables full of the type of :class:`Bone` used
    for that table. If you're selecting data from a table, the
    :class:`Bone`s should be filled in with the keys you want, but
    their other fields should be set to ``None``. You may need to do
    this explicitly, if a field has some default value other than
    ``None``.

    To get the particular :class:`Bone` subclass for a given table of
    a given class, access the ``bonetypes`` attribute of the
    class. For instance, suppose there is a table named ``ham`` in a
    class named ``Spam``, and you want to select the records with the
    field ``eggs`` equal to 3, or ``beans`` equal to True:

    ``Spam._select_skeleton(cursor, {u"ham":
    [Spam.bonetypes.ham(eggs=3), Spam.bonetypes.ham(beans=True)]})``

    This expression will return a :class:`Skeleton` with all the
    results in. The outermost layer of the :class:`Skeleton` will be
    keyed with the table names, in this case just "ham"; successive
    layers are each keyed with one of the fields in the primary key of
    the table.

    The same :class:`Skeleton` may later be passed to
    :method:`_insert_skeleton`, presumably with some of the data
    changed.


    Dependencies and Custom SQL
    ===========================

    The LiSE database schema uses a lot of foreign keys, which only
    work if they refer to a table that exists. To make sure it does,
    the class that uses a foreign key will have the names of the
    tables it points to in a class attribute called ``demands``. The
    tables demanded are looked up in other classes' attribute called
    ``provides`` to ensure they've been taken care of. Both attributes
    are :type:`set`s.``provides`` is generated automatically, but will
    accept any additions you give it, which is occasionally necessary
    when a class deals with SQL that is not generated in the usual
    way.

    If you want to do something with SQL that
    :class:`SaveableMetaclass` does not do for you, put the SQL
    statements you want executed in :type:`list` attributes on the
    classes, called ``prelude`` and ``postlude``. All statements in
    ``prelude`` will be executed prior to declaring the first table in
    the class. All statements in ``postlude`` will be executed after
    declaring the last table in the class. In both cases, they are
    executed in the order of iteration, so use a sequence type.

    """
    def __new__(metaclass, clas, parents, attrs):
        """Return a new class with all the accoutrements of
        :class:`SaveableMetaclass`.

        """
        global schemata
        global tabclas
        tablenames = []
        "Names of tables declared in the ``tables`` attribute of ``clas``."
        foreignkeys = {}
        """Keys: table names. Values: dictionaries with keys of table names
        *linked to*, values of tuples of field names linked to."""
        coldecls = {}
        """Unprocessed column declarations, the 1th item of a tuple in a
        ``tables`` attribute."""
        checks = {}
        """For each table, a list of Boolean expressions meant for a
        CHECK(...) clause."""
        if 'tables' in attrs:
            tabdicts = attrs['tables']
        else:
            return type.__new__(metaclass, clas, parents, attrs)
        if 'prelude' in attrs:
            prelude = attrs["prelude"]
        else:
            prelude = []
        if 'postlude' in attrs:
            postlude = attrs["postlude"]
        else:
            postlude = []
        if 'provides' in attrs:
            provides = set(attrs["provides"])
        else:
            provides = set()
        if 'demands' in attrs:
            demands = set(attrs["demands"])
        else:
            demands = set()
        local_pkeys = {}
        """Primary keys per-table, for this class only"""
        for (name, tabdict) in tabdicts:
            tablenames.append(name)
            coldecls[name] = tabdict["columns"]
            local_pkeys[name] = tabdict["primary_key"]
            primarykeys[name] = tabdict["primary_key"]
            if "foreign_keys" in tabdict:
                foreignkeys[name] = tabdict["foreign_keys"]
            else:
                foreignkeys[name] = {}
            if "checks" in tabdict:
                checks[name] = tabdict["checks"]
            else:
                checks[name] = []
        inserts = {}
        """The beginnings of SQL INSERT statements"""
        deletes = {}
        """The beginnings of SQL DELETE statements"""
        keylen = {}
        rowlen = {}
        keyqms = {}
        rowqms = {}
        keystrs = {}
        rowstrs = {}
        keynames = {}
        valnames = {}
        coltypes = {}
        coldefaults = {}
        bonetypes = {}
        for item in local_pkeys.items():
            (tablename, pkey) = item
            keynames[tablename] = list(pkey)
            keylen[tablename] = len(pkey)
            keyqms[tablename] = ", ".join(["?"] * keylen[tablename])
            keystrs[tablename] = "(" + keyqms[tablename] + ")"
        for item in coldecls.items():
            (tablename, coldict) = item
            coltypes[tablename] = {}
            coldefaults[tablename] = {}
            for (fieldname, decl) in coldict.iteritems():
                if fieldname == "branch":
                    foreignkeys[tablename][fieldname] = (
                        "timestream", "branch")
                    checks[tablename].append("branch>=0")
                cooked = decl.split(" ")
                typename = cooked[0]
                coltypes[tablename][fieldname] = {
                    "text": unicode,
                    "int": int,
                    "integer": int,
                    "bool": bool,
                    "boolean": bool,
                    "float": float}[typename.lower()]
                try:
                    default_str = cooked[cooked.index("default") + 1]
                    default = coltypes[tablename][fieldname](default_str)
                except ValueError:
                    default = None
                coldefaults[tablename][fieldname] = default
            valnames[tablename] = list(set(coldict.keys()) -
                                       set(keynames[tablename]))
            rowlen[tablename] = len(coldict)
            rowqms[tablename] = ", ".join(["?"] * rowlen[tablename])
            rowstrs[tablename] = "(" + rowqms[tablename] + ")"
        for tablename in coldecls.keys():
            colnames[tablename] = keynames[tablename] + valnames[tablename]
        for tablename in tablenames:
            assert(tablename not in tabclas)
            bonetypes[tablename] = Bone.subclass(
                tablename,
                [(colname,
                  coltypes[tablename][colname],
                  coldefaults[tablename][colname])
                 for colname in colnames[tablename]])
            tabclas[tablename] = clas
            provides.add(tablename)
            coldecl = coldecls[tablename]
            pkey = primarykeys[tablename]
            fkeys = foreignkeys[tablename]
            cks = ["CHECK(%s)" % ck for ck in checks[tablename]]
            coldecs = [coln + " " + coldecl[coln]
                       for coln in colnames[tablename]]
            coldecstr = ", ".join(coldecs)
            pkeycolstr = ", ".join(pkey)
            colnamestr[tablename] = ", ".join(colnames[tablename])
            pkeystr = "PRIMARY KEY (%s)" % (pkeycolstr,)
            fkeystrs = []
            for item in fkeys.items():
                if len(item[1]) == 2:
                    fkeystrs.append(
                        "FOREIGN KEY (%s) REFERENCES %s(%s)" %
                        (item[0], item[1][0], item[1][1]))
                elif len(item[1]) == 3:
                    fkeystrs.append(
                        "FOREIGN KEY ({0}) REFERENCES {1}({2}) {3}".format(
                            item[0], item[1][0], item[1][1], item[1][2]))
                else:
                    raise Exception("Invalid foreign key: {0}".format(item))
            fkeystr = ", ".join(fkeystrs)
            chkstr = ", ".join(cks)
            table_decl_data = [coldecstr]
            if len(pkey) > 0:
                table_decl_data.append(pkeystr)
            if len(fkeystrs) > 0:
                table_decl_data.append(fkeystr)
            if len(cks) > 0:
                table_decl_data.append(chkstr)
            table_decl = ", ".join(table_decl_data)
            create_stmt = "CREATE TABLE %s (%s)" % (tablename, table_decl)
            insert_stmt_start = ("INSERT INTO " + tablename +
                                 " ({0}) VALUES {1};")
            inserts[tablename] = insert_stmt_start
            delete_stmt_start = "DELETE FROM %s WHERE (%s) IN " % (
                tablename, pkeycolstr)
            deletes[tablename] = delete_stmt_start
            schemata[tablename] = create_stmt
        saveables.append(
            (tuple(demands),
             tuple(provides),
             tuple(prelude),
             tuple(tablenames),
             tuple(postlude)))

        def gen_sql_insert(tabname):
            """Return an SQL INSERT statement suitable for adding one row, with
            all the fields filled in."""
            return "INSERT INTO {0} ({1}) VALUES {2};".format(
                tabname,
                colnamestr[tabname],
                rowstrs[tabname])

        @staticmethod
        def insert_bones_table(c, bones, tabname):
            """Use the cursor ``c` to insert the bones into the table
            ``tabname``."""
            try:
                c.executemany(gen_sql_insert(tabname), bones)
            except IntegrityError as ie:
                print(ie)
                print(gen_sql_insert(tabname))

        def gen_sql_delete(keybone, tabname):
            """Return an SQL DELETE statement to get rid of the record
            corresponding to ``keybone`` in table ``tabname``."""
            try:
                keyns = keynames[tabname]
            except KeyError:
                return
            keys = []
            if tabname not in tablenames:
                raise ValueError("Unknown table: {}".format(tabname))
            checks = []
            for keyn in keyns:
                checks.append(keyn + "=?")
                keys.append(getattr(keybone, keyn))
            where = "(" + " AND ".join(checks) + ")"
            qrystr = "DELETE FROM {0} WHERE {1}".format(tabname, where)
            return (qrystr, tuple(keys))

        @staticmethod
        def delete_keybones_table(c, keybones, tabname):
            """Use the cursor ``c`` to delete the records matching the
            bones from the table ``tabname``.

            """
            for keybone in keybones:
                c.execute(*gen_sql_delete(keybone, tabname))

        def gen_sql_select(keybones, tabname):
            """Return an SQL SELECT statement to get the records
            matching the bones from the table ``tabname``.

            """
            # Assumes that all keybones have the same type.
            together = []
            for bone in keybones:
                apart = []
                for field in bone._fields:
                    if getattr(bone, field) is not None:
                        apart.append("{}=?".format(field))
                together.append("({})".format(" AND ".join(apart)))
            orstr = " OR ".join(together)
            return "SELECT {0} FROM {1} WHERE {2};".format(
                ", ".join(keybones[0]._fields), tabname, orstr)

        def select_keybones_table(c, keybones, tabname):
            """Return a list of records taken from the table ``tabname``,
            through the cursor ``c``, matching the bones.

            """
            qrystr = gen_sql_select(keybones, tabname)
            qrylst = []
            for bone in keybones:
                for key in bone._fields:
                    if getattr(bone, key) is not None:
                        qrylst.append(getattr(bone, key))
            if len(qrylst) == 0:
                return []
            c.execute(qrystr, tuple(qrylst))
            return c.fetchall()

        @staticmethod
        def _select_skeleton(c, skel):
            """Return a new :class:`Skeleton` like ``skel``, but with the bones
            filled in with live data. Requires a cursor ``c``.

            ``skel`` needs to have table names for its keys. It may be a
            :type:`dict` instead of a :class:`Skeleton`.

            """
            r = Skeleton({})
            for (tabname, bones) in skel.items():
                if tabname not in r:
                    r[tabname] = {}
                for row in select_keybones_table(c, bones, tabname):
                    bone = bonetypes[tabname](*row)
                    ptr = r[tabname]
                    for key in bone.cls.keynames[bone._name][:-1]:
                        if getattr(bone, key) not in ptr:
                            ptr[getattr(bone, key)] = {}
                        ptr = ptr[getattr(bone, key)]
                    finkey = bone.cls.keynames[bone._name][-1]
                    ptr[getattr(bone, finkey)] = bone
            return r

        @staticmethod
        def _insert_skeleton(c, skeleton):
            """Use the cursor ``c`` to insert the bones in the
            skeleton into the table given by the key.

            """
            for (tabname, rds) in skeleton.items():
                if tabname in tablenames:
                    insert_bones_table(c, rds, tabname)

        @staticmethod
        def _delete_skeleton(c, skeleton):
            """Use the cursor ``c`` to delete records from the tables in
            the skeleton's keys, when they match any of the bones in the
            skeleton's values.

            """
            for (tabname, records) in skeleton.items():
                if tabname in tablenames:
                    bones = [bone for bone in records.iterbones()]
                    delete_keybones_table(c, bones, tabname)

        atrdic = {
            '_select_skeleton': _select_skeleton,
            '_insert_skeleton': _insert_skeleton,
            '_delete_skeleton': _delete_skeleton,
            '_insert_bones_table': insert_bones_table,
            '_delete_keybones_table': delete_keybones_table,
            '_gen_sql_insert': gen_sql_insert,
            '_gen_sql_delete': gen_sql_delete,
            'colnames': dict([
                (tabn, tuple(colnames[tabn])) for tabn in tablenames]),
            'colnamestr': dict([
                (tabn, unicode(colnamestr[tabn])) for tabn in tablenames]),
            'colnstr': colnamestr[tablenames[0]],
            'keynames': dict([
                (tabn, tuple(keynames[tabn])) for tabn in tablenames]),
            'valnames': dict([
                (tabn, tuple(valnames[tabn])) for tabn in tablenames]),
            'keyns': tuple(keynames[tablenames[0]]),
            'valns': tuple(valnames[tablenames[0]]),
            'colns': tuple(colnames[tablenames[0]]),
            'keylen': keylen,
            'rowlen': rowlen,
            'keyqms': keyqms,
            'rowqms': rowqms,
            'maintab': tablenames[0],
            'tablenames': tuple(tablenames),
            'bonetypes': bonetypes,
            'bonetype': bonetypes[tablenames[0]]}
        atrdic.update(attrs)

        clas = type.__new__(metaclass, clas, parents, atrdic)
        saveable_classes.append(clas)
        for bonetype in bonetypes.itervalues():
            bonetype.cls = clas
        return clas


def slope_theta_rise_run(rise, run):
    """Return a radian value expressing the angle at the lower-left corner
    of a triangle ``rise`` high, ``run`` wide.

    If ``run`` is zero, but ``rise`` is positive, return pi / 2. If
    ``run`` is zero, but ``rise`` is negative, return -pi / 2.

    """
    try:
        return atan(rise/run)
    except ZeroDivisionError:
        if rise >= 0:
            return ninety
        else:
            return -1 * ninety


def slope_theta(ox, oy, dx, dy):
    """Get a radian value representing the angle formed at the corner (ox,
    oy) of a triangle with a hypotenuse going from there to (dx,
    dy).

    """
    rise = dy - oy
    run = dx - ox
    return slope_theta_rise_run(rise, run)


def opp_theta_rise_run(rise, run):
    """Inverse of ``slope_theta_rise_run``"""
    try:
        return atan(run/rise)
    except ZeroDivisionError:
        if run >= 0:
            return ninety
        else:
            return -1 * ninety


def opp_theta(ox, oy, dx, dy):
    """Inverse of ``slope_theta``"""
    rise = dy - oy
    run = dx - ox
    return opp_theta_rise_run(rise, run)


def truncated_line(leftx, boty, rightx, topy, r, from_start=False):
    """Return coordinates for two points, very much like the two points
    supplied, but with the end of the line foreshortened by amount r.

    """
    # presumes pointed up and right
    if r == 0:
        return (leftx, boty, rightx, topy)
    rise = topy - boty
    run = rightx - leftx
    length = hypot(rise, run) - r
    theta = slope_theta_rise_run(rise, run)
    if from_start:
        leftx = rightx - cos(theta) * length
        boty = topy - sin(theta) * length
    else:
        rightx = leftx + cos(theta) * length
        topy = boty + sin(theta) * length
    return (leftx, boty, rightx, topy)


def wedge_offsets_core(theta, opp_theta, taillen):
    """Internal use"""
    top_theta = theta - fortyfive
    bot_theta = pi - fortyfive - opp_theta
    xoff1 = cos(top_theta) * taillen
    yoff1 = sin(top_theta) * taillen
    xoff2 = cos(bot_theta) * taillen
    yoff2 = sin(bot_theta) * taillen
    return (
        xoff1, yoff1, xoff2, yoff2)


def wedge_offsets_rise_run(rise, run, taillen):
    """Given a line segment's rise, run, and length, return two new
    points--with respect to the *end* of the line segment--that are good
    for making an arrowhead with.

    The arrowhead is a triangle formed from these points and the point at
    the end of the line segment.

    """
    # theta is the slope of a line bisecting the ninety degree wedge.
    theta = slope_theta_rise_run(rise, run)
    opp_theta = opp_theta_rise_run(rise, run)
    return wedge_offsets_core(theta, opp_theta, taillen)


ninety = pi / 2
"""pi / 2"""

fortyfive = pi / 4
"""pi / 4"""


class TimestreamException(Exception):
    """Used for time travel related errors that are nothing to do with
continuity."""
    pass


class TimeParadox(Exception):
    """I tried to record some fact at some time, and in so doing,
    contradicted the historical record."""
    pass


class JourneyException(Exception):
    """There was a problem with pathfinding."""
    pass


class KnowledgeException(Exception):
    """I tried to access some information that I was not permitted access to.

    Should be treated like KeyError most of the time. For the purposes
    of the simulation, not having information is the same as
    information not existing. But there may be circumstances where
    they differ for programming purposes.

    """
    pass


class ListItemIterator:
    """Iterate over a list in a way that resembles dict.iteritems()"""
    def __init__(self, l):
        """Initialize for list l"""
        self.l = l
        self.l_iter = iter(l)
        self.i = 0

    def __iter__(self):
        """I'm an iterator"""
        return self

    def __len__(self):
        """Provide the length of the underlying list."""
        return len(self.l)

    def __next__(self):
        """Return a tuple of the current index and its item in the list"""
        it = next(self.l_iter)
        i = self.i
        self.i += 1
        return (i, it)

    def next(self):
        """Return a tuple of the current index and its item in the list"""
        return self.__next__()


class Fabulator(object):
    """Construct objects (or call functions, as you please) as described
    by strings loaded in from the database.

    This doesn't use exec(). You need to supply the functions when you
    construct the Fabulator.

    """
    def __init__(self, fabs):
        """Supply a dictionary full of callables, keyed by the names you want
        to use for them.

        """
        self.fabbers = fabs

    def __call__(self, s):
        """Parse the string into something I can make a callable from. Then
        make it, using the classes in self.fabbers.

        """
        (outer, inner) = match("(.+)\((.+)\)", s).groups()
        return self._call_recursively(outer, inner)

    def _call_recursively(self, outer, inner):
        """Internal use"""
        fun = self.fabbers[outer]
        # pretty sure parentheses are meaningless inside []
        m = findall("(.+)\((.+)\)[,)] *", inner)
        if len(m) == 0:
            return fun(*inner.split(",").strip(" "))
        elif len(m) == 1:
            (infun, inarg) = m[0]
            infun = self.fabbers[infun]
            inargs = inarg.split(",").strip(" ")
            return fun(infun(*inargs))
        else:
            # This doesn't allow any mixing of function-call arguments
            # with text arguments at the same level. Not optimal.
            return fun(*[self._call_recursively(infun, inarg)
                         for (infun, inarg) in m])
