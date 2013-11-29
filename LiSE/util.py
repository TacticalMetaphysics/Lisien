# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from math import sqrt, hypot, atan, pi, sin, cos
from sqlite3 import IntegrityError
from collections import (
    MutableMapping,
    OrderedDict)
from operator import itemgetter
from re import match, compile, findall
import struct


"""Common utility functions and data structures.

The most important are Skeleton, a mapping used to store and maintain
all game data; and SaveableMetaclass, which generates
SQL from metadata declared as class atttributes.

"""

phi = (1.0 + sqrt(5))/2.0

schemata = {}

colnames = {}

colnamestr = {}

primarykeys = {}

tabclas = {}

saveables = []

saveable_classes = []


thingex = compile("Thing\((.+?)\)")

placex = compile("Place\((.+?)\)")

portex = compile("Portal\((.+?)->(.+?)\)")


def get_bone_during(skel, branch, tick):
    """Convenience function for looking up the current effective value of
something in a Skeleton.

    The current effective value is the latest one that took effect at
    or prior to the present tick.

    """
    if branch not in skel:
        return None
    ikeys = set(skel[branch].ikeys)
    tick_from = 0
    test_tick = ikeys.pop()
    while tick_from != tick and len(ikeys) > 0:
        if tick_from < test_tick < tick:
            tick_from = test_tick
        test_tick = ikeys.pop()
    try:
        return skel[branch][tick_from]
    except KeyError:
        return None


class BoneMetaclass(type):
    def __new__(metaclass, clas, parents, attrs):
        # Presently there's no way to distinguish between "I have not
        # decided what to put here" and "This database record has a
        # null value in this field". I think the only cases where this
        # presents trouble are if you want an update rather than an
        # insert (updates aren't supported) or if you want to query
        # for nulls in some field, which I suppose to be a strange
        # enough case to ignore.
        def __new__(_cls, *args, **kwargs):
            """Create new instance of {}""".format(clas)
            if len(args) > 0:
                return tuple.__new__(_cls, args)
            values = []
            for fieldn in _cls._fields:
                if fieldn in kwargs:
                    if type(kwargs[fieldn]) not in (
                            _cls._types[fieldn], type(None)):
                        kwargs[fieldn] = _cls._types[fieldn](kwargs[fieldn])
                    values.append(kwargs[fieldn])
                else:
                    values.append(_cls._defaults[fieldn])
            return tuple.__new__(_cls, tuple(values))

        @classmethod
        def _make(cls, iterable):
            """Make a new {} object from a sequence or iterable""".format(
                clas)
            return tuple.__new__(cls, iterable)

        @classmethod
        def getfmt(cls):
            fmt = bytearray('@')
            for (field_name, field_type, default) in cls._field_decls:
                if field_type in (unicode, str):
                    fmt.extend('50s')
                elif field_type is int:
                    fmt.append('l')
                elif field_type is float:
                    fmt.append('d')
                else:
                    raise TypeError(
                        "Trying to make a format string; "
                        "don't understand the type "
                        "{}".format(field_type))
            return str(fmt)

        @property
        def packed(self):
            """Return a string of data packed according to self.format."""
            args = [self.getfmt()]
            for datum in self:
                if isinstance(datum, unicode):
                    args.append(str(datum))
                else:
                    args.append(datum)
            return struct.pack(*args)

        @classmethod
        def _unpack(cls, data):
            """Return a new instance of this class using the packed data in the
string.

            """
            r = cls(*struct.unpack(cls.getfmt(), data))
            denulled = {}
            for field in r._fields:
                if isinstance(getattr(r, field), str):
                    denulled[field] = unicode(
                        getattr(r, field)).replace('\x00', '')
            return r._replace(**denulled)

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
                "getfmt": getfmt,
                "packed": packed,
                "_unpack": _unpack}
        atts.update(attrs)
        i = 0
        if "_fields" in atts:
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
    __metaclass__ = BoneMetaclass
    _field_decls = {}

    @classmethod
    def subclass(cls, name, decls):
        return type(name, (cls,), {"_field_decls": decls})


class Skeleton(MutableMapping):
    """A tree structure similar to a database.

When all my keys are integers, iteration over my children will proceed
in the order of those integers.

The + and - operators are interpreted in a way similar to set union
and set difference.

There's a limited sort of event handler triggered by __setitem__ and
__delitem__. Append listeners to self.listeners.

    """
    def __init__(self, content=None, name="", parent=None):
        """Technically all of the arguments are optional but you should really
specify them whenever it's reasonably sensible to do so. content is a
mapping to crib from, parent is who to propagate events to, and name
is mostly for printing."""
        self.listeners = []
        self.ikeys = set([])
        self.name = name
        self.parent = parent
        self.content = {}
        if content is not None:
            self._populate_content(content)

    def _populate_content(self, content):
        if content in (None, {}, []):
            return
        assert(not issubclass(content.__class__, Bone))
        if isinstance(content, dict):
            kitr = content.iteritems()
        elif isinstance(content, self.__class__):
            kitr = content.iteritems()
        else:
            kitr = ListItemIterator(content)
        for (k, v) in kitr:
            if hasattr(v, 'content'):
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
        if isinstance(self.content, dict):
            return k in self.content
        else:
            return k in self.ikeys

    def __getitem__(self, k):
        if isinstance(self.content, list) and k not in self.ikeys:
            raise KeyError("key not in skeleton: {}".format(k))
        return self.content[k]

    def __setitem__(self, k, v):
        if hasattr(self, 'bones_only'):
            if v is not None and not issubclass(v.__class__, Bone):
                raise TypeError(
                    "A Skeleton directly containing Bones "
                    "can't directly contain anything else.")
        if len(self.content) == 0:
            self.ikeys = set([])
            if isinstance(k, int):
                self.content = []
        if isinstance(k, int):
            while len(self.content) <= k:
                self.content.append(None)
            self.ikeys.add(k)
        if isinstance(v, self.__class__):
            v.name = k
            v.parent = self
            self.content[k] = v
        elif issubclass(v.__class__, Bone):
            self.bones_only = True
            self.content[k] = v
        else:
            self.content[k] = self.__class__(v, name=k, parent=self)
        for listener in self.listeners:
            listener(self, k, v)
        if hasattr(self.parent, 'on_child_set'):
            self.parent.on_child_set(self, k, v)

    def on_child_set(self, child, k, v):
        for listener in self.listeners:
            listener(child, k, v)
        if hasattr(self.parent, 'on_child_set'):
            self.parent.on_child_set(child, k, v)

    def __delitem__(self, k):
        if isinstance(self.content, dict):
            del self.content[k]
        else:
            self.ikeys.remove(k)
        for listener in self.listeners:
            listener(self, k)
        if hasattr(self.parent, 'on_child_del'):
            self.parent.on_child_del(self, k)

    def on_child_del(self, child, k):
        for listener in self.listeners:
            listener(child, k)
        if hasattr(self.parent, 'on_child_del'):
            self.parent.on_child_del(child, k)

    def __iter__(self):
        if isinstance(self.content, dict):
            return iter(self.content)
        else:
            return iter(sorted(self.ikeys))

    def __len__(self):
        if isinstance(self.content, dict):
            return len(self.content)
        else:
            return len(self.ikeys)

    def __repr__(self):
        return repr(self.content)

    def __iadd__(self, other):
        self.update(other)
        return self

    def __add__(self, other):
        selfie = self.copy()
        selfie.update(other)
        return selfie

    def __isub__(self, other):
        for (k, v) in other.items():
            if k not in self:
                continue
            elif v == self[k]:
                del self[k]
            elif isinstance(self[k], Skeleton) and isinstance(v, Skeleton):
                self[k] -= v
            # otherwise, just keep it
        return self

    def __sub__(self, other):
        selfie = self.copy()
        selfie -= other
        return selfie

    def __str__(self):
        return str(self.name)

    def keys(self):
        if isinstance(self.content, dict):
            return self.content.keys()
        else:
            return sorted(self.ikeys)

    def key_before(self, k):
        if hasattr(self, 'ikeys'):
            ikeys = set(self.ikeys)
            afore = None
            while len(ikeys) > 0 and afore != k - 1:
                ik = ikeys.pop()
                if ik < k:
                    if afore is None or ik > afore:
                        afore = ik
            return afore
        return max([(j for j in self.content.keys() if j < k)])

    def key_or_key_before(self, k):
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
        return self[self.key_or_key_before(k)]

    def key_after(self, k):
        if hasattr(self, 'ikeys'):
            ikeys = set(self.ikeys)
            aft = None
            while len(ikeys) > 0 and aft != k + 1:
                ik = ikeys.pop()
                if (aft is None and ik > k) or (
                        k < ik < aft):
                    aft = ik
            return aft
        return min([(j for j in self.content.keys() if j > k)])

    def key_or_key_after(self, k):
        if hasattr(self, 'ikeys'):
            if k in self.ikeys:
                return k
            else:
                return self.key_after(k)
        if k in self.content:
            return k
        else:
            return self.key_after(k)

    def bone_at_or_after(self, k):
        return self[self.key_or_key_after(k)]

    def copy(self):
        if isinstance(self.content, list):
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
                else:
                    assert(False)
            return self.__class__(content=r)

    def deepcopy(self):
        r = {}
        for (k, v) in self.content.items():
            r[k] = v.deepcopy()
        return self.__class__(
            content=r, name=self.name, parent=self.parent)

    def update(self, skellike):
        for (k, v) in skellike.iteritems():
            if issubclass(v.__class__, Bone):
                self[k] = v
            elif isinstance(v, self.__class__):
                if k in self.content:
                    self[k].update(v)
                else:
                    self[k] = v.copy()
                    self[k].parent = self
                    self[k].name = k
            else:
                if k in self.content:
                    self[k].update(v)
                else:
                    assert(v.__class__ in (dict, list))
                    self[k] = self.__class__(
                        content=v, name=k, parent=self)

    def iterbones(self):
        if isinstance(self.content, dict):
            for contained in self.content.itervalues():
                if issubclass(contained.__class__, Bone):
                    yield contained
                else:
                    for bone in contained.iterbones():
                        yield bone
        else:
            for i in sorted(self.ikeys):
                if issubclass(self.content[i].__class__, Bone):
                    yield self.content[i]
                else:
                    for bone in self.content[i].iterbones():
                        yield bone

    def get_closet(self):
        ptr = self.parent
        while isinstance(ptr, self.__class__):
            ptr = ptr.parent
        return ptr


class SaveableMetaclass(type):
    """Sort of an object relational mapper.

Classes with this metaclass need to be declared with an attribute
called tables. This is a sequence of tuples. Each of the tuples is of
length 5. Each describes a table that records what's in the class.

The meaning of each tuple is thus:

(name, column_declarations, primary_key, foreign_keys, checks)

name is the name of the table as sqlite3 will use it.

column_declarations is a dictionary. The keys are field names, aka
column names. Each value is the type for its field, perhaps including
a clause like DEFAULT 0.

primary_key is an iterable over strings that are column names as
declared in the previous argument. Together the columns so named form
the primary key for this table.

foreign_keys is a dictionary. Each foreign key is a key here, and its
value is a pair. The first element of the pair is the foreign table
that the foreign key refers to. The second element is the field or
fields in that table that the foreign key points to.

checks is an iterable over strings that will end up in a CHECK(...)
clause in sqlite3.

A class can have any number of such table-tuples. The tables will be
declared in the order they appear in the tables attribute.

    """
    def __new__(metaclass, clas, parents, attrs):
        global schemata
        global tabclas
        tablenames = []
        foreignkeys = {}
        coldecls = {}
        checks = {}
        if 'tables' in attrs:
            tablist = attrs['tables']
        elif hasattr(clas, 'tables'):
            tablist = clas.tables
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
        for tabtup in tablist:
            (name, decls, pkey, fkeys, cks) = tabtup
            tablenames.append(name)
            coldecls[name] = decls
            local_pkeys[name] = pkey
            primarykeys[name] = pkey
            foreignkeys[name] = fkeys
            checks[name] = cks
        inserts = {}
        deletes = {}
        detects = {}
        missings = {}
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
            keynames[tablename] = sorted(pkey)
            keylen[tablename] = len(pkey)
            keyqms[tablename] = ", ".join(["?"] * keylen[tablename])
            keystrs[tablename] = "(" + keyqms[tablename] + ")"
        for item in coldecls.items():
            (tablename, coldict) = item
            coltypes[tablename] = {}
            coldefaults[tablename] = {}
            for (fieldname, decl) in coldict.iteritems():
                cooked = decl.lower().split(" ")
                typename = cooked[0]
                coltypes[tablename][fieldname] = {
                    "text": unicode,
                    "int": int,
                    "integer": int,
                    "bool": bool,
                    "boolean": bool,
                    "float": float}[typename]
                try:
                    default_str = cooked[cooked.index("default") + 1]
                    default = coltypes[tablename][fieldname](default_str)
                except ValueError:
                    default = None
                coldefaults[tablename][fieldname] = default
            valnames[tablename] = sorted(
                [key for key in list(coldict.keys())
                 if key not in keynames[tablename]])
            rowlen[tablename] = len(coldict)
            rowqms[tablename] = ", ".join(["?"] * rowlen[tablename])
            rowstrs[tablename] = "(" + rowqms[tablename] + ")"
        for tablename in coldecls.keys():
            colnames[tablename] = keynames[tablename] + valnames[tablename]
        for tablename in tablenames:
            assert(tablename not in tabclas)
            bonetypes[tablename] = type(
                tablename + "_bone",
                (Bone,),
                {"_field_decls": [
                    (colname,
                     coltypes[tablename][colname],
                     coldefaults[tablename][colname])
                    for colname in colnames[tablename]]})
            tabclas[tablename] = clas
            provides.add(tablename)
            coldecl = coldecls[tablename]
            pkey = primarykeys[tablename]
            fkeys = foreignkeys[tablename]
            cks = ["CHECK(%s)" % ck for ck in checks[tablename]]
            pkeydecs = [keyname + " " + typ
                        for (keyname, typ) in coldecl.items()
                        if keyname in pkey]
            valdecs = [valname + " " + typ
                       for (valname, typ) in coldecl.items()
                       if valname not in pkey]
            coldecs = sorted(pkeydecs) + sorted(valdecs)
            coldecstr = ", ".join(coldecs)
            pkeycolstr = ", ".join(pkey)
            pkeys = [keyname for (keyname, typ) in coldecl.items()
                     if keyname in pkey]
            pkeynamestr = ", ".join(sorted(pkeys))
            vals = [valname for (valname, typ) in coldecl.items()
                    if valname not in pkey]
            colnamestr[tablename] = ", ".join(sorted(pkeys) + sorted(vals))
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
            create_stmt = "CREATE TABLE %s (%s);" % (tablename, table_decl)
            insert_stmt_start = ("INSERT INTO " + tablename +
                                 " ({0}) VALUES {1};")
            inserts[tablename] = insert_stmt_start
            delete_stmt_start = "DELETE FROM %s WHERE (%s) IN " % (
                tablename, pkeycolstr)
            deletes[tablename] = delete_stmt_start
            detect_stmt_start = "SELECT %s FROM %s WHERE (%s) IN " % (
                colnamestr[tablename], tablename, pkeynamestr)
            detects[tablename] = detect_stmt_start
            missing_stmt_start = "SELECT %s FROM %s WHERE (%s) NOT IN " % (
                colnamestr[tablename], tablename, pkeynamestr)
            missings[tablename] = missing_stmt_start
            schemata[tablename] = create_stmt
        saveables.append(
            (tuple(demands),
             tuple(provides),
             tuple(prelude),
             tuple(tablenames),
             tuple(postlude)))

        def gen_sql_insert(bones, tabname):
            varlst = []
            qrylst = []
            some_data = False
            for bone in bones:
                varlst.append(rowstrs[tabname])
                qrylst.extend([getattr(bone, coln) for coln in
                               colnames[tabname]])
                some_data = True
            if not some_data:
                raise ValueError("No data to insert.")
            qrystr = "INSERT INTO {0} ({1}) VALUES {2}".format(
                tabname,
                colnamestr[tabname],
                ", ".join(varlst))
            return (qrystr, tuple(qrylst))

        @staticmethod
        def insert_bones_table(c, bones, tabname):
            try:
                c.execute(*gen_sql_insert(bones, tabname))
            except IntegrityError as ie:
                print(ie)
                print(gen_sql_insert(bones, tabname))
            except EmptySkeleton:
                return

        def gen_sql_delete(keybone, tabname):
            try:
                keyns = keynames[tabname]
            except KeyError:
                return
            keys = []
            if tabname not in tablenames:
                raise EmptySkeleton
            checks = []
            for keyn in keyns:
                checks.append(keyn + "=?")
                keys.append(getattr(keybone, keyn))
            where = "(" + " AND ".join(checks) + ")"
            qrystr = "DELETE FROM {0} WHERE {1}".format(tabname, where)
            return (qrystr, tuple(keys))

        @staticmethod
        def delete_keybones_table(c, keybones, tabname):
            for keybone in keybones:
                try:
                    c.execute(*gen_sql_delete(keybone, tabname))
                except EmptySkeleton:
                    return

        def gen_sql_select(keybones, tabname):
            # Assumes that all keybones have the same type.
            keys = []
            for k in primarykeys[tabname]:
                if getattr(keybones[0], k) is not None:
                    keys.append(k)
            andstr = "({0})".format(
                " AND ".join(
                    ["{0}=?".format(key) for key in keys]
                ))
            ands = [andstr] * len(keybones)
            colstr = colnamestr[tabname]
            orstr = " OR ".join(ands)
            return "SELECT {0} FROM {1} WHERE {2}".format(
                colstr, tabname, orstr)

        def select_keybones_table(c, keybones, tabname):
            keys = primarykeys[tabname]
            qrystr = gen_sql_select(keybones, tabname)
            qrylst = []
            for bone in keybones:
                for key in keys:
                    if getattr(bone, key) is not None:
                        qrylst.append(getattr(bone, key))
            if len(qrylst) == 0:
                return []
            c.execute(qrystr, tuple(qrylst))
            return c.fetchall()

        @staticmethod
        def _select_table_all(c, tabname):
            r = Skeleton({})
            qrystr = "SELECT {0} FROM {1}".format(
                colnamestr[tabname], tabname)
            c.execute(qrystr)
            for row in c.fetchall():
                bone = row2bone(row, getattr(bonetypes, tabname))
                ptr = r
                oldptr = None
                unset = True
                for key in primarykeys[tabname]:
                    if getattr(bone, key) not in ptr:
                        try:
                            ptr[getattr(bone, key)] = Skeleton()
                        except TypeError:
                            ptr[getattr(bone, key)] = bone
                            unset = False
                            break
                    oldptr = ptr
                    ptr = ptr[getattr(bone, key)]
                if unset:
                    oldptr[getattr(bone, key)] = bone
                # rd = dictify_row(row, colnames[tabname])
                # ptr = r
                # lptr = r
                # for key in primarykeys[tabname]:
                #     if rd[key] not in ptr:
                #         ptr[rd[key]] = {}
                #     lptr = ptr
                #     ptr = ptr[rd[key]]
                # lptr[rd[key]] = rd
            return Skeleton({tabname: r})

        @staticmethod
        def _select_skeleton(c, td):
            r = Skeleton({})
            for (tabname, bones) in td.items():
                if tabname not in primarykeys:
                    continue
                if tabname not in r:
                    r[tabname] = {}
                for row in select_keybones_table(c, bones, tabname):
                    bone = row2bone(row, getattr(bonetypes, tabname))
                    ptr = r[tabname]
                    oldptr = None
                    unset = True
                    for key in primarykeys[tabname]:
                        if getattr(bone, key) not in ptr:
                            try:
                                ptr[getattr(bone, key)] = {}
                            except TypeError:
                                ptr[getattr(bone, key)] = bone
                                unset = False
                                break
                        oldptr = ptr
                        ptr = ptr[getattr(bone, key)]
                    if unset:
                        oldptr[getattr(bone, key)] = bone
                    # rd = dictify_row(row, colnames[tabname])
                    # ptr = r[tabname]
                    # keys = list(primarykeys[tabname])
                    # oldptr = None
                    # while keys != []:
                    #     key = keys.pop(0)
                    #     if rd[key] not in ptr:
                    #         ptr[rd[key]] = {}
                    #     oldptr = ptr
                    #     ptr = ptr[rd[key]]
                    # oldptr[rd[key]] = rd
            return r

        def gen_sql_detect(bonedict, tabname):
            keystr = keystrs[tabname]
            qrystr = detects[tabname] + ", ".join([keystr] * len(bonedict))
            qrylst = []
            for bone in bonedict:
                if not isinstance(bone, bonetypes[tabname]):
                    raise TypeError(
                        "{} is the wrong bone type for {}".format(
                            type(bone), tabname))
                for field in keynames[tabname]:
                    if field in bonetypes[tabname]._fields:
                        qrylst.append(getattr(bone, field))
            qrytup = tuple(qrylst)
            return (qrystr, qrytup)

        def detect_keydicts_table(c, keydicts, tabname):
            c.execute(*gen_sql_detect(keydicts, tabname))
            return c.fetchall()

        def gen_sql_missing(keydicts, tabname):
            keystr = keystrs[tabname]
            qrystr = missings[tabname] + ", ".join([keystr] * len(keydicts))
            qrylst = []
            for keydict in iter(keydicts):
                for col in keynames[tabname]:
                    if col in keydict:
                        qrylst.append(keydict[col])
            qrytup = tuple(qrylst)
            return (qrystr, qrytup)

        def missing_keydicts_table(c, keydicts, tabname):
            c.execute(*gen_sql_missing(keydicts, tabname))
            return c.fetchall()

        @staticmethod
        def _insert_skeleton(c, skeleton):
            for (tabname, rds) in skeleton.items():
                if tabname in tablenames:
                    insert_bones_table(c, rds, tabname)

        @staticmethod
        def _delete_skeleton(c, skeleton):
            for (tabname, records) in skeleton.items():
                if tabname in tablenames:
                    bones = [bone for bone in records.iterbones()]
                    delete_keybones_table(c, bones, tabname)

        @staticmethod
        def _detect_skeleton(c, skeleton):
            r = {}
            for item in skeleton.items():
                (tabname, rd) = item
                if isinstance(rd, dict):
                    r[tabname] = detect_keydicts_table(c, [rd], tabname)
                else:
                    r[tabname] = detect_keydicts_table(c, rd, tabname)
            return r

        @staticmethod
        def _missing_skeleton(c, skeleton):
            r = {}
            for item in skeleton.items():
                (tabname, rd) = item
                if isinstance(rd, dict):
                    r[tabname] = missing_keydicts_table(c, [rd], tabname)
                else:
                    r[tabname] = missing_keydicts_table(c, rd, tabname)
            return r

        def get_keydict(self):
            tabd = self.get_skeleton()
            r = {}
            for tabn in tablenames:
                r[tabn] = {}
                for keyn in keynames[tabn]:
                    r[tabn][keyn] = tabd[tabn][keyn]
            return r

        bonetypes = type(clas + '_bonetypes',
                         (Bone,),
                         {'_field_decls': [
                             (tabn, type(bonetypes[tabn]), bonetypes[tabn])
                             for tabn in tablenames]})()
        atrdic = {
            '_select_skeleton': _select_skeleton,
            '_select_table_all': _select_table_all,
            '_insert_skeleton': _insert_skeleton,
            '_delete_skeleton': _delete_skeleton,
            '_detect_skeleton': _detect_skeleton,
            '_missing_skeleton': _missing_skeleton,
            '_insert_bones_table': insert_bones_table,
            '_delete_keybones_table': delete_keybones_table,
            '_gen_sql_insert': gen_sql_insert,
            '_gen_sql_delete': gen_sql_delete,
            '_gen_sql_detect': gen_sql_detect,
            '_gen_sql_missing': gen_sql_missing,
            'colnames': type(
                clas + '_colnames',
                (Bone,),
                {'_field_decls': [
                    (tabn, tuple, tuple(colnames[tabn]))
                    for tabn in tablenames]})(),
            'colnamestr': type(
                clas + '_colnamestr',
                (Bone,),
                {'_field_decls': [
                    (tabn, unicode, unicode(colnamestr[tabn]))]})(),
            'colnstr': colnamestr[tablenames[0]],
            'keynames': Bone.subclass(
                clas + '_keynames',
                [(tabn, tuple, tuple(keynames[tabn]))
                 for tabn in tablenames])(),
            'valnames': Bone.subclass(
                clas + '_valnames',
                [(tabn, tuple, tuple(valnames[tabn]))
                 for tabn in tablenames])(),
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
            'bonetype': bonetypes[0]}
        atrdic.update(attrs)

        clas = type.__new__(metaclass, clas, parents, atrdic)
        saveable_classes.append(clas)
        return clas


def start_new_map(nope):
    pass


def open_map(nope):
    pass


def save_map(nope):
    pass


def quit_map_editor(nope):
    pass


def editor_select(nope):
    pass


def editor_copy(nope):
    pass


def editor_paste(nope):
    pass


def editor_delete(nope):
    pass


def new_place(place_type):
    pass


def new_thing(thing_type):
    pass


funcs = [start_new_map, open_map, save_map, quit_map_editor, editor_select,
         editor_copy, editor_paste, editor_delete, new_place, new_thing]


def keyify_dict(d, keytup):
    ptr = d
    for key in keytup:
        if key not in ptr:
            ptr[key] = {}
        ptr = ptr[key]


def dictify_row(row, colnames):
    return dict(list(zip(colnames, row)))


def row2bone(row, bonetype):
    return bonetype(**dict(zip(bonetype._fields, row)))


def deep_lookup(dic, keylst):
    key = keylst.pop()
    ptr = dic
    while keylst != []:
        ptr = ptr[key]
        key = keylst.pop()
    return ptr[key]


def stringlike(o):
    """Return True if I can easily cast this into a string, False
otherwise."""
    return isinstance(o, str) or isinstance(o, str)


def place2idx(db, dimname, pl):
    if isinstance(pl, int):
        return pl
    elif hasattr(pl, '_index'):
        return pl._index
    elif stringlike(pl):
        try:
            return int(pl)
        except ValueError:
            return db.placedict[str(dimname)][pl].i
    else:
        raise ValueError("Can't convert that into a place-index")


def line_len(ox, oy, dx, dy):
    rise = dy - oy
    run = dx - ox
    return hypot(rise, run)


def slope_theta_rise_run(rise, run):
    try:
        return atan(rise/run)
    except ZeroDivisionError:
        if rise >= 0:
            return ninety
        else:
            return -1 * ninety


def slope_theta(ox, oy, dx, dy):
    rise = dy - oy
    run = dx - ox
    return slope_theta_rise_run(rise, run)


def opp_theta_rise_run(rise, run):
    try:
        return atan(run/rise)
    except ZeroDivisionError:
        if run >= 0:
            return ninety
        else:
            return -1 * ninety


def opp_theta(ox, oy, dx, dy):
    rise = dy - oy
    run = dx - ox
    return opp_theta_rise_run(rise, run)


def truncated_line(leftx, boty, rightx, topy, r, from_start=False):
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


def extended_line(leftx, boty, rightx, topy, r):
    return truncated_line(leftx, boty, rightx, topy, -1 * r)


def trimmed_line(leftx, boty, rightx, topy, trim_start, trim_end):
    et = truncated_line(leftx, boty, rightx, topy, trim_end)
    return truncated_line(et[0], et[1], et[2], et[3], trim_start, True)


def wedge_offsets_core(theta, opp_theta, taillen):
    top_theta = theta - fortyfive
    bot_theta = pi - fortyfive - opp_theta
    xoff1 = cos(top_theta) * taillen
    yoff1 = sin(top_theta) * taillen
    xoff2 = cos(bot_theta) * taillen
    yoff2 = sin(bot_theta) * taillen
    return (
        xoff1, yoff1, xoff2, yoff2)


def wedge_offsets_rise_run(rise, run, taillen):
    # theta is the slope of a line bisecting the ninety degree wedge.
    theta = slope_theta_rise_run(rise, run)
    opp_theta = opp_theta_rise_run(rise, run)
    return wedge_offsets_core(theta, opp_theta, taillen)


def wedge_offsets_slope(slope, taillen):
    theta = atan(slope)
    opp_theta = atan(1/slope)
    return wedge_offsets_core(theta, opp_theta, taillen)


def average(*args):
    n = len(args)
    return sum(args)/n


ninety = pi / 2

fortyfive = pi / 4

threesixty = pi * 2


class BranchTicksIter:
    def __init__(self, d):
        self.branchiter = iter(d.items())
        self.branch = None
        self.tickfromiter = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            (tick_from, vtup) = next(self.tickfromiter)
            if isinstance(vtup, tuple):
                tick_to = vtup[-1]
                value = vtup[:-1]
                return (self.branch, tick_from, tick_to) + value
            else:
                return (self.branch, tick_from, vtup)
        except (AttributeError, StopIteration):
            (self.branch, tickfromdict) = next(self.branchiter)
            self.tickfromiter = iter(tickfromdict.items())
            return next(self)


class PortalException(Exception):
    """Exception raised when a Thing tried to move into or out of or along
a Portal, and it made no sense."""
    pass


class LocationException(Exception):
    pass


class ContainmentException(Exception):
    """Exception raised when a Thing tried to go into or out of another
Thing, and it made no sense.

    """
    pass


class LoadError(Exception):
    pass


class TimestreamException(Exception):
    pass


class TimeParadox(Exception):
    pass


class JourneyException(Exception):
    pass


class ListItemIterator:
    """Iterate over a list in a way that resembles dict.iteritems()"""
    def __init__(self, l):
        self.l = l
        self.l_iter = iter(l)
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.l)

    def __next__(self):
        it = next(self.l_iter)
        i = self.i
        self.i += 1
        return (i, it)

    def next(self):
        return self.__next__()


class FilterIter:
    def __init__(self, itr, do_not_return):
        self.real = iter(itr)
        self.do_not_return = do_not_return

    def __iter__(self):
        return self

    def __next__(self):
        r = next(self.real)
        while r in self.do_not_return:
            r = next(self.real)
        return r

    def next(self):
        return self.__next__()


class FirstOfTupleFilter:
    def __init__(self, containable):
        self.containable = containable

    def __contains__(self, t):
        return t[0] in self.containable


class FakeCloset:
    def __init__(self, skellike):
        self.skeleton = Skeleton(skellike)
        self.timestream = Timestream(self)


class BranchError(Exception):
    pass


def next_val_iter(litter):
    try:
        if len(litter) <= 1:
            raise StopIteration
        return litter[:-1] + [iter(litter[-2].next().values())]
    except StopIteration:
        if len(litter) <= 1:
            raise StopIteration
        nvi = next_val_iter(litter[:-1])
        return nvi + [iter(nvi[-1].next().values())]


def skel_nth_generator(skel, n):
    iters = [iter(skel.values())]
    for i in range(0, n-1):
        iters.append(iter(next(iters[-1]).values()))
    try:
        yield next(iters[-1])
    except StopIteration:
        if len(iters) <= 1:
            raise StopIteration
        iters = next_val_iter(iters)
        yield next(iters[-1])


class Timestream(object):
    __metaclass__ = SaveableMetaclass
    # I think updating the start and end ticks of a branch using
    # listeners might be a good idea
    tables = [
        ("timestream",
         {"branch": "integer not null",
          "parent": "integer not null"},
         ("branch",),
         {"parent": ("timestream", "branch")},
         ["branch>=0", "parent=0 or parent<>branch"])
        ]

    listen_tables = set([
        "thing_location"])

    tab_depth = {
        "character_places": 3,
        "character_portals": 4,
        "character_things": 3,
        "character_skills": 2,
        "character_stats": 2,
        "pawn_img": 3,
        "pawn_interactive": 3,
        "portal": 3,
        "spot_coords": 3,
        "spot_img": 3,
        "spot_interactive": 3,
        "thing_location": 2}

    def __init__(self, closet):
        self.closet = closet
        self.hi_branch_listeners = []
        self.hi_tick_listeners = []
        self.hi_branch = 0
        self.hi_tick = 0

    def __setattr__(self, attrn, val):
        if attrn == "hi_branch":
            self.set_hi_branch(val)
        elif attrn == "hi_tick":
            self.set_hi_tick(val)
        else:
            super(Timestream, self).__setattr__(attrn, val)

    def set_hi_branch(self, b):
        for listener in self.hi_branch_listeners:
            listener(self, b)
        super(Timestream, self).__setattr__("hi_branch", b)

    def set_hi_tick(self, t):
        for listener in self.hi_tick_listeners:
            listener(self, t)
        super(Timestream, self).__setattr__("hi_tick", t)

    def branches(self, table=None):
        if table is None:
            return self.allbranches()
        else:
            return self.branchtable(table)

    def branchtable(self, table):
        n = self.tab_depth[table]
        if n == 1:
            for d in self.closet.skeleton[table].values():
                for k in d.keys():
                    yield k
        else:
            for d in skel_nth_generator(self.closet.skeleton[table], n):
                for k in d.keys():
                    yield k

    def allbranches(self):
        for (tabn, n) in self.tab_depth.items():
            if n == 1:
                for d in self.closet.skeleton[tabn].values():
                    for k in d.keys():
                        yield k
            else:
                try:
                    for d in skel_nth_generator(
                            self.closet.skeleton[tabn], n):
                        for k in d.keys():
                            yield k
                except TypeError:
                    yield 0
                    return

    def allticks(self):
        for (tabn, n) in self.tab_depth.items():
                if n == 1:
                    for d in self.closet.skeleton[tabn].values():
                        for k in d.keys():
                            yield k
                else:
                    try:
                        for d in skel_nth_generator(
                                self.closet.skeleton[tabn], n):
                            for k in d.keys():
                                yield k
                    except TypeError:
                        yield 0
                        return

    def branchticks(self, branch):
        for (tabn, n) in self.tab_depth.items():
            if n == 1:
                for d in self.closet.skeleton[tabn].values():
                    for k in d[branch].keys():
                        yield k
            else:
                n = self.tab_depth[tabn]
                ptr = self.closet.skeleton[tabn]
                try:
                    for i in range(0, n):
                        ptr = next(iter(ptr.values()))
                except StopIteration:
                    continue
                for k in ptr[branch].keys():
                    yield k

    def tabticks(self, table):
        n = self.tab_depth[table]
        if n == 1:
            for d in self.closet.skeleton[table]:
                for k in d.keys():
                    yield k
        else:
            for d in skel_nth_generator(self.closet.skeleton[table], n):
                for k in d.keys():
                    yield k

    def branchtabticks(self, branch, table):
        n = self.tab_depth[table]
        if n == 1:
            for k in self.closet.skeleton[table][branch].keys():
                yield k
        else:
            ptr = self.closet.skeleton[table]
            for i in range(0, n):
                ptr = next(iter(ptr.values()))
            if branch in ptr:
                for k in ptr[branch].keys():
                    yield k

    def ticks(self, branch=None, table=None):
        if branch is None and table is None:
            return self.allticks()
        elif table is None:
            return self.branchticks(branch)
        elif branch is None:
            return self.tabticks(table)
        else:
            return self.branchtabticks(branch, table)

    def max_branch(self, table=None):
        r = max(self.branches(table))
        return r

    def min_branch(self, table=None):
        return min(self.branches(table))

    def max_tick(self, branch=None, table=None):
        try:
            return max(self.ticks(branch, table))
        except (KeyError, ValueError):
            return None

    def min_tick(self, branch=None, table=None):
        try:
            return min(self.ticks(branch, table))
        except (KeyError, ValueError):
            return None

    def uptick(self, tick):
        self.hi_tick = max((tick, self.hi_tick))

    def upbranch(self, branch):
        self.hi_branch = max((branch, self.hi_branch))

    def parent(self, branch):
        assert(branch > 0)
        return self.closet.skeleton["timestream"][branch]["parent"]

    def children(self, branch):
        for bone in self.closet.skeleton["timestream"].iterbones():
            if bone.parent == branch:
                yield bone.branch


class EmptySkeleton(Exception):
    pass


class Fabulator(object):
    """Construct objects (or call functions, as you please) as described
by strings loaded in from the database. exec()-free.

    """
    def __init__(self, fabs):
        """Supply a dictionary full of callables, keyed by the names you want
to use for them."""
        self.fabbers = fabs

    def __call__(self, s):
        """Parse the string into something I can make a callable from. Then
make it, using the classes in self.fabbers."""
        (outer, inner) = match("(.+)\((.+)\)", s).groups()
        return self.call_recursively(outer, inner)

    def call_recursively(self, outer, inner):
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
            return fun(*[self.call_recursively(infun, inarg)
                         for (infun, inarg) in m])
