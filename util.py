# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import pyglet
import ctypes
from time import time
from math import sqrt, hypot, atan, pi, sin, cos
from logging import getLogger
from sqlite3 import IntegrityError
from igraph import Graph, Vertex, Edge
from collections import deque, defaultdict, MutableMapping
from copy import copy, deepcopy

logger = getLogger(__name__)

phi = (1.0 + sqrt(5))/2.0

schemata = {}

colnames = {}

colnamestr = {}

primarykeys = {}

tabclas = {}

saveables = []

saveable_classes = []


class Skeleton(MutableMapping):
    """A dict-like object to efficiently store the entire world model.

Each LiSE instance should have one Skeleton in one Closet. Every
object in that instance is a view onto the Skeleton.

To get a particular object's data, first decide what table you want
data from; that table's name is the key in the top level of the
Skeleton. Then look it up in the Skeleton using the same keys as you
would to get it from the database. Each successive field in the
primary key gets its own level of the Skeleton.

I say "the" Skeleton, but actually every level of the Skeleton is
another Skeleton, and may be addressed similarly.

The skeleton's leaves are "rowdicts": dictionaries keyed with field
names, whose values are the values of those fields in a given row. You
can look up the whole primary key from just the rowdict.

Apart from the usual iterkeys(), itervalues(), and iteritems(), the
Skeleton has a special iterator method iterrows(). This does a
depth-first traversal through the Skeleton and gives you each rowdict
it comes upon."""
    atrdic = {
        "typ": lambda self: {
            True: lambda: type(self.it),
            False: lambda: None}[hasattr(self, 'it')](),
        "subtype": lambda self: self.getsubtype(),
        "rowdict": lambda self: self.isrowdict()
        }

    def __init__(self, it, parent=None, listeners=None):
        self.parent = parent
        if listeners is None:
            self.listeners = set()
        else:
            self.listeners = set(listeners)
        if isinstance(it, Skeleton):
            self.it = copy(it.it)
            self.parent = it.parent
        elif isinstance(it, list):
            self.it = []
            for that in it:
                assert(that.__class__ in (Skeleton, dict, list))
                if isinstance(that, Skeleton):
                    self.it.append(that)
                else:
                    self.it.append(Skeleton(that, parent=self))
        elif isinstance(it, dict):
            rowdict = False
            self.it = {}
            for (k, v) in it.iteritems():
                if v.__class__ not in (Skeleton, dict, list):
                    rowdict = True
                if rowdict:
                    assert(v.__class__ not in (Skeleton, dict, list))
                    self.it[k] = v
                else:
                    self.it[k] = Skeleton(v, parent=self)
        else:
            assert False

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Skeleton instance does not have and cannot compute "
                "attribute {0}".format(attrn))

    def __getitem__(self, k):
        if isinstance(self.it, list):
            if k < len(self.it):
                return self.it[k]
            elif self.subtype is None:
                raise KeyError(
                    "That part of the skeleton is empty "
                    "and I can't decide how to fill it. "
                    "If you want to help out, add something.")
            while len(self.it) <= k:
                self.it.append(Skeleton(self.subtype(), parent=self))
            return self.it[k]
        elif isinstance(self.it, dict):
            if k in self.it:
                return self.it[k]
            elif self.subtype is None:
                raise KeyError(
                    "That part of the skeleton is empty "
                    "and I can't decide how to fill it. "
                    "If you want to help out, add something.")
            else:
                self.it[k] = Skeleton(self.subtype(), parent=self)
                return self.it[k]
        else:
            assert(False)

    def __setitem__(self, k, v):
        if self.typ is list:
            if type(k) is not int:
                raise TypeError(
                    "This level of the Skeleton requires integer keys.")
            if self.subtype is not None:
                newtyp = self.subtype
            elif isinstance(v, Skeleton):
                newtyp = v.typ
            elif isinstance(v, list):
                newtyp = list
            elif isinstance(v, dict):
                newtyp = dict
            else:
                raise ValueError(
                    "This level of the Skeleton can only "
                    "hold Skeleton-alikes.")
            i = 0
            while len(self.it) <= k:
                self.it.append(Skeleton(newtyp(), parent=self))
                i += 1
        elif self.rowdict:
            assert(v.__class__ not in (list, dict, Skeleton))
            self.it[k] = v
            for listener in self.listeners:
                listener.on_skel_set((self,), k, v)
            if self.parent is not None:
                self.parent.on_child_set((self,), k, v)
            return
        nuval = Skeleton(v, parent=self)
        self.it[k] = nuval
        for listener in self.listeners:
            listener.on_skel_set((self,), k, nuval)
        if self.parent is not None:
            self.parent.on_child_set((self,), k, nuval)

    def __delitem__(self, k):
        if self.typ is list:
            self[k] = self.subtype()
        else:
            del self.it[k]
            for listener in self.listeners:
                listener.on_skel_delete((self,), k)
        if self.parent is not None:
            self.parent.on_child_delete((self,), k)

    def __contains__(self, what):
        if self.typ is list:
            return what > -1 and what < len(self.it)
        else:
            return what in self.it

    def __iter__(self):
        if self.typ is list:
            return iter(xrange(0, len(self.it)))
        else:
            return self.it.iterkeys()

    def __len__(self):
        return len(self.it)

    def __add__(self, other):
        newness = self.copy()
        newness += other.copy()
        return newness

    def __iadd__(self, other):
        self.update(other)
        return self

    def __sub__(self, other):
        newness = self.copy()
        newness -= other.copy()
        return newness

    def __isub__(self, other):
        assert(isinstance(other, Skeleton))
        kitr = other.iteritems()
        for (k, v) in kitr:
            if k not in self:
                continue
            elif v == self[k]:
                assert(isinstance(self[k], Skeleton))
                assert(isinstance(v, Skeleton))
                del self[k]
            elif self[k].rowdict:
                continue
            else:
                assert(isinstance(self[k], Skeleton))
                assert(isinstance(v, Skeleton))
                self[k] -= v
        return self

    def __repr__(self):
        return "Skeleton({0})".format(repr(self.it))

    def __eq__(self, other):
        if isinstance(other, Skeleton):
            return self.it == other.it
        elif isinstance(other, dict):
            return self.typ is dict and self.it == other
        else:
            return self.typ is list and self.it == other

    def __dict__(self):
        r = {}
        for (k, v) in self.iteritems():
            if isinstance(v, Skeleton):
                r[k] = v.__dict__()
            else:
                r[k] = v
        return r

    def copy(self):
        # Sort of a deep copy but doesn't contain any lineage. This is
        # actually what I want when I'm doing "set difference,"
        # because then I'm not deleting anything from the original,
        # and I don't want any listeners to get that impression.
        return Skeleton(deepcopy(self.__dict__()))

    def deepcopy(self):
        # This one DOES contain the lineage
        if self.rowdict:
            return Skeleton(self.it, self.parent, self.listeners)
        elif self.typ is list:
            it = [inner.deepcopy() for inner in self.it]
            return Skeleton(it, self.parent, self.listeners)
        else:
            it = {}
            for (k, v) in self.it.iteritems():
                it[k] = v.deepcopy()
            return Skeleton(it, self.parent, self.listeners)

    def iteritems(self):
        if self.typ is list:
            return ListItemIterator(self.it)
        else:
            return self.it.iteritems()

    def iterrows(self):
        return SkeletonIterator(self)

    def keys(self):
        if self.typ is list:
            return [
                i for i in xrange(0, len(self.it))
                if self.it[i] is not None]
        else:
            return self.it.keys()

    def isrowdict(self):
        if len(self.it) == 0:
            return False
        for v in self.itervalues():
            if v.__class__ in (Skeleton, dict, list):
                return False
        return True

    def getsubtype(self):
        if self.rowdict or len(self.it) == 0:
            return None
        if self.typ is dict:
            return self.it.itervalues().next().typ
        else:  # self.typ is list
            return self.it[0].typ

    def update(self, skellike):
        if skellike.__class__ in (dict, Skeleton):
            kitr = skellike.iteritems()
        else:
            kitr = ListItemIterator(skellike)
        for (k, v) in kitr:
            if v.__class__ in (dict, list):
                v = Skeleton(v, parent=self)
                if v.rowdict:
                    self[k] = v
                    continue
            elif self.rowdict or self.it == {}:
                self.it[k] = v
                continue
            if not isinstance(v, Skeleton):
                continue
            if isinstance(self.it, dict):
                if k in self.it:
                    self.it[k].update(v)
                else:
                    self.it[k] = v
            else:
                if self.subtype is not None:
                    newtyp = self.subtype
                else:
                    newtyp = v.typ
                i = 0
                while len(self.it) <= k:
                    self.it.append(Skeleton(newtyp(), parent=self))
                    i += 1
                if isinstance(self.it[k], Skeleton):
                    self.it[k].update(v)
                else:
                    self.it[k] = v

    def on_child_set(self, child, k, v):
        chain = (self, child)
        for listener in self.listeners:
            listener.on_skel_set(chain, k, v)
        if self.parent is not None:
            self.parent.on_child_set(chain, k, v)

    def on_child_delete(self, child, k):
        chain = (self, child)
        for listener in self.listeners:
            listener.on_skel_delete(chain, k)
        if self.parent is not None:
            self.parent.on_child_delete(chain, k)

    def add_listener(self, l):
        logger.debug("Adding listener: {0} To skeleton: {1}".format(l, self))
        self.listeners.add(l)


class SaveableMetaclass(type):
# TODO make savers use sets of RowDict objs, rather than lists of regular dicts
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

To save, you need to define a method called get_skeleton. It should
return a dictionary where the keys are table names. The values are
either rowdicts or iterables over rowdicts. A rowdict is a dictionary
containing the information in a single record of a table; the keys are
the names of the fields.

To load, you need to define a method called from_skeleton that takes
that same kind of dictionary and returns an instance of your class.

Once you've defined those, the save(db) and load(db) methods will save
or load your class in the given database. If you need to create the
database, look at the schemata attribute: execute that in a SQL cursor
and your table will be ready.

    """
    def __new__(metaclass, clas, parents, attrs):
        global schemata
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
        for item in local_pkeys.iteritems():
            (tablename, pkey) = item
            keynames[tablename] = sorted(pkey)
            keylen[tablename] = len(pkey)
            keyqms[tablename] = ", ".join(["?"] * keylen[tablename])
            keystrs[tablename] = "(" + keyqms[tablename] + ")"
        for item in coldecls.iteritems():
            (tablename, coldict) = item
            valnames[tablename] = sorted(
                [key for key in coldict.keys()
                 if key not in keynames[tablename]])
            rowlen[tablename] = len(coldict)
            rowqms[tablename] = ", ".join(["?"] * rowlen[tablename])
            rowstrs[tablename] = "(" + rowqms[tablename] + ")"
        for tablename in coldecls.iterkeys():
            colnames[tablename] = keynames[tablename] + valnames[tablename]
        for tablename in tablenames:
            provides.add(tablename)
            coldecl = coldecls[tablename]
            pkey = primarykeys[tablename]
            fkeys = foreignkeys[tablename]
            cks = ["CHECK(%s)" % ck for ck in checks[tablename]]
            pkeydecs = [keyname + " " + typ
                        for (keyname, typ) in coldecl.iteritems()
                        if keyname in pkey]
            valdecs = [valname + " " + typ
                       for (valname, typ) in coldecl.iteritems()
                       if valname not in pkey]
            coldecs = sorted(pkeydecs) + sorted(valdecs)
            coldecstr = ", ".join(coldecs)
            pkeycolstr = ", ".join(pkey)
            pkeys = [keyname for (keyname, typ) in coldecl.iteritems()
                     if keyname in pkey]
            pkeynamestr = ", ".join(sorted(pkeys))
            vals = [valname for (valname, typ) in coldecl.iteritems()
                    if valname not in pkey]
            colnamestr[tablename] = ", ".join(sorted(pkeys) + sorted(vals))
            pkeystr = "PRIMARY KEY (%s)" % (pkeycolstr,)
            fkeystrs = []
            for item in fkeys.iteritems():
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
            (demands, provides, prelude, tablenames, postlude))

        def gen_sql_insert(rowdicts, tabname):
            if tabname in rowdicts:
                itr = SkeletonIterator(rowdicts[tabname])
            else:
                itr = SkeletonIterator(rowdicts)
            if len(itr) == 0 or tabname not in tablenames:
                raise EmptyTabdict
            qrystr = "INSERT INTO {0} ({1}) VALUES {2}".format(
                tabname,
                colnamestr[tabname],
                ", ".join([rowstrs[tabname]] * len(itr)))
            qrylst = []
            for rd in itr:
                qrylst.extend([rd[coln] for coln in colnames[tabname]])
            return (qrystr, tuple(qrylst))

        @staticmethod
        def insert_rowdicts_table(c, rowdicts, tabname):
            if len(rowdicts) == 0:
                return []
            try:
                c.execute(*gen_sql_insert(rowdicts, tabname))
            except IntegrityError as ie:
                print ie
                print gen_sql_insert(rowdicts, tabname)
                import pdb
                pdb.set_trace()
            except EmptyTabdict:
                return

        def gen_sql_delete(keydicts, tabname):
            try:
                keyns = keynames[tabname]
            except KeyError:
                return
            keys = []
            wheres = []
            kitr = SkeletonIterator(keydicts)
            if len(kitr) == 0 or tabname not in tablenames:
                raise EmptyTabdict
            for keydict in kitr:
                checks = []
                for keyn in keyns:
                    checks.append(keyn + "=?")
                    try:
                        keys.append(keydict[keyn])
                    except KeyError as ke:
                        print ke
                        import pdb
                        pdb.set_trace()
                wheres.append("(" + " AND ".join(checks) + ")")
            wherestr = " OR ".join(wheres)
            qrystr = "DELETE FROM {0} WHERE {1}".format(tabname, wherestr)
            return (qrystr, tuple(keys))

        @staticmethod
        def delete_keydicts_table(c, keydicts, tabname):
            if len(keydicts) == 0:
                return
            try:
                c.execute(*gen_sql_delete(keydicts, tabname))
            except EmptyTabdict:
                return

        def gen_sql_select(keydicts, tabname):
            keys_in_use = set()
            kitr = SkeletonIterator(keydicts)
            for keyd in kitr:
                for k in keyd:
                    keys_in_use.add(k)
            keys = [key for key in primarykeys[tabname] if key in keys_in_use]
            andstr = "({0})".format(
                " AND ".join(
                    ["{0}=?".format(key) for key in keys]
                ))
            ands = [andstr] * len(kitr)
            colstr = colnamestr[tabname]
            orstr = " OR ".join(ands)
            return "SELECT {0} FROM {1} WHERE {2}".format(
                colstr, tabname, orstr)

        def select_keydicts_table(c, keydicts, tabname):
            keys = primarykeys[tabname]
            qrystr = gen_sql_select(keydicts, tabname)
            qrylst = []
            kitr = SkeletonIterator(keydicts)
            for keydict in kitr:
                for key in keys:
                    try:
                        qrylst.append(keydict[key])
                    except KeyError:
                        pass
            if len(qrylst) == 0:
                return []
            c.execute(qrystr, tuple(qrylst))
            return c.fetchall()

        @staticmethod
        def _select_table_all(c, tabname):
            r = {}
            qrystr = "SELECT {0} FROM {1}".format(
                colnamestr[tabname], tabname)
            c.execute(qrystr)
            for row in c.fetchall():
                rd = dictify_row(row, colnames[tabname])
                ptr = r
                lptr = r
                for key in primarykeys[tabname]:
                    if rd[key] not in ptr:
                        ptr[rd[key]] = {}
                    lptr = ptr
                    ptr = ptr[rd[key]]
                lptr[rd[key]] = rd
            return {tabname: r}

        @staticmethod
        def _select_skeleton(c, td):
            r = Skeleton({})
            for (tabname, rdd) in td.iteritems():
                if tabname not in primarykeys:
                    continue
                if tabname not in r:
                    r[tabname] = {}
                for row in select_keydicts_table(c, rdd, tabname):
                    rd = dictify_row(row, colnames[tabname])
                    ptr = r[tabname]
                    keys = list(primarykeys[tabname])
                    oldptr = None
                    while keys != []:
                        key = keys.pop(0)
                        if rd[key] not in ptr:
                            ptr[rd[key]] = {}
                        oldptr = ptr
                        ptr = ptr[rd[key]]
                    oldptr[rd[key]] = rd
            return r

        def gen_sql_detect(keydicts, tabname):
            keystr = keystrs[tabname]
            qrystr = detects[tabname] + ", ".join([keystr] * len(keydicts))
            qrylst = []
            for keydict in iter(keydicts):
                for col in keynames[tabname]:
                    if col in keydict:
                        qrylst.append(keydict[col])
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
            for (tabname, rds) in skeleton.iteritems():
                if tabname in tablenames:
                    insert_rowdicts_table(c, rds, tabname)

        @staticmethod
        def _delete_skeleton(c, skeleton):
            for (tabname, rds) in skeleton.iteritems():
                if tabname in tablenames:
                    delete_keydicts_table(c, rds, tabname)

        @staticmethod
        def _detect_skeleton(c, skeleton):
            r = {}
            for item in skeleton.iteritems():
                (tabname, rd) = item
                if isinstance(rd, dict):
                    r[tabname] = detect_keydicts_table(c, [rd], tabname)
                else:
                    r[tabname] = detect_keydicts_table(c, rd, tabname)
            return r

        @staticmethod
        def _missing_skeleton(c, skeleton):
            r = {}
            for item in skeleton.iteritems():
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

        atrdic = {
            '_select_skeleton': _select_skeleton,
            '_select_table_all': _select_table_all,
            '_insert_skeleton': _insert_skeleton,
            '_delete_skeleton': _delete_skeleton,
            '_detect_skeleton': _detect_skeleton,
            '_missing_skeleton': _missing_skeleton,
            '_insert_rowdicts_table': insert_rowdicts_table,
            '_delete_keydicts_table': delete_keydicts_table,
            '_gen_sql_insert': gen_sql_insert,
            '_gen_sql_delete': gen_sql_delete,
            '_gen_sql_detect': gen_sql_detect,
            '_gen_sql_missing': gen_sql_missing,
            'colnames': colnames,
            'colnamestr': colnamestr,
            'colnstr': colnamestr[tablenames[0]],
            'keynames': keynames,
            'valnames': valnames,
            'keyns': keynames[tablenames[0]],
            'valns': valnames[tablenames[0]],
            'colns': colnames[tablenames[0]],
            'keylen': keylen,
            'rowlen': rowlen,
            'keyqms': keyqms,
            'rowqms': rowqms,
            'maintab': tablenames[0],
            'tablenames': tablenames}
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


def tickly_get(db, get_from, branch, tick):
    if branch is None:
        branch = db.branch
    if tick is None:
        tick = db.tick
    if branch not in get_from:
        return None
    if tick in get_from[branch]:
        return get_from[branch][tick]
    for (tick_from, (val, tick_to)) in get_from[branch].iteritems():
        if tick_from <= tick and tick <= tick_to:
            return val
    return None


def untuple(list_o_tups):
    r = []
    for tup in list_o_tups:
        for val in tup:
            r.append(val)
    return r


def dictify_row(row, colnames):
    return dict(zip(colnames, row))


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
    return isinstance(o, str) or isinstance(o, unicode)


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


def get_line_width():
    see = ctypes.c_float()
    pyglet.gl.glGetFloatv(pyglet.gl.GL_LINE_WIDTH, see)
    return float(see.value)


def set_line_width(w):
    wcf = ctypes.c_float(w)
    pyglet.gl.glLineWidth(wcf)


def average(*args):
    n = len(args)
    return sum(args)/n


ninety = pi / 2

fortyfive = pi / 4

threesixty = pi * 2


class BranchTicksIter:
    def __init__(self, d):
        self.branchiter = d.iteritems()
        self.branch = None
        self.tickfromiter = None

    def __iter__(self):
        return self

    def next(self):
        try:
            (tick_from, vtup) = self.tickfromiter.next()
            if isinstance(vtup, tuple):
                tick_to = vtup[-1]
                value = vtup[:-1]
                return (self.branch, tick_from, tick_to) + value
            else:
                return (self.branch, tick_from, vtup)
        except (AttributeError, StopIteration):
            (self.branch, tickfromdict) = self.branchiter.next()
            self.tickfromiter = tickfromdict.iteritems()
            return self.next()


class TerminableImg:
    __metaclass__ = SaveableMetaclass

    def get_img(self, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if len(self.imagery) < branch:
            return None
        if branch in self.indefinite_imagery:
            indef_start = self.indefinite_imagery[branch]
            if tick >= indef_start:
                rd = self.imagery[branch][indef_start]
                return self.closet.get_img(rd["img"])
        for rd in SkeletonIterator(self.imagery[branch]):
            if rd["tick_from"] <= tick and tick <= rd["tick_to"]:
                return self.closet.get_img(rd["img"])
        return None

    def new_branch_imagery(self, parent, branch, tick):
        for rd in SkeletonIterator(self.imagery[parent]):
            if rd["tick_to"] is None or rd["tick_to"] >= tick:
                rd2 = dict(rd)
                rd2["branch"] = branch
                if rd2["tick_from"] < tick:
                    rd2["tick_from"] = tick
                    self.imagery[branch][tick] = rd2
                    if rd2["tick_to"] is None:
                        self.indefinite_imagery[branch] = tick
                else:
                    self.imagery[branch][rd["tick_from"]] = rd2
                    if rd2["tick_to"] is None:
                        self.indefinite_imagery[branch] = rd2["tick_from"]


class TerminableInteractivity:
    __metaclass__ = SaveableMetaclass

    def is_interactive(self, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if branch not in self.interactivity:
            return False
        if (
                branch in self.indefinite_interactivity and
                tick >= self.indefinite_interactivity[branch]):
            return True
        for rd in SkeletonIterator(self.interactivity):
            if rd["tick_from"] <= tick and tick <= rd["tick_to"]:
                return True
        return False

    def new_branch_interactivity(self, parent, branch, tick):
        for rd in SkeletonIterator(self.interactivity[parent]):
            if rd["tick_to"] is None or rd["tick_to"] >= tick:
                rd2 = dict(rd)
                rd2["branch"] = branch
                if rd2["tick_from"] < tick:
                    rd2["tick_from"] = tick
                    self.interactivity[branch][tick] = rd2
                else:
                    self.interactivity[branch][rd2["tick_from"]] = rd2
                if rd2["tick_to"] is None:
                    self.indefinite_interactivity[branch] = rd2["tick_from"]


class ViewportOrderedGroup(pyglet.graphics.OrderedGroup):
    def __init__(self, order, parent, view):
        super(ViewportOrderedGroup, self).__init__(order, parent)
        self.view = view

    def __getattr__(self, attrn):
        oddtype = pyglet.gl.gl.GLint * 4
        r = oddtype()
        pyglet.gl.gl.glGetIntegerv(pyglet.gl.GL_VIEWPORT, r)
        if attrn == "width":
            return r[2]
        elif attrn == "height":
            return r[3]
        else:
            raise AttributeError(
                "ViewportOrderedGroup has no attribute " + attrn)

    def set_state(self):
        tup = (
            self.view.window_left,
            self.view.window_bot,
            self.view.width,
            self.view.height)
        pyglet.gl.glViewport(*tup)
        pyglet.gl.glScissor(*tup)
        pyglet.gl.glEnable(pyglet.gl.GL_SCISSOR_TEST)

    def unset_state(self):
        pyglet.gl.glViewport(
            0,
            0,
            self.view.window.width,
            self.view.window.height)
        pyglet.gl.glDisable(pyglet.gl.GL_SCISSOR_TEST)


class PatternHolder:
    """Takes a style and makes pyglet.image.SolidColorImagePatterns out of
its four colors, accessible through the attributes bg_active,
bg_inactive, fg_active, and fg_inactive."""
    def __init__(self, sty):
        self.bg_inactive = (
            pyglet.image.SolidColorImagePattern(sty.bg_inactive.tup))
        self.bg_active = (
            pyglet.image.SolidColorImagePattern(sty.bg_active.tup))
        self.fg_inactive = (
            pyglet.image.SolidColorImagePattern(sty.fg_inactive.tup))
        self.fg_active = pyglet.image.SolidColorImagePattern(sty.fg_active.tup)


class DictValues2DIterator:
    def __init__(self, d):
        self.d = d
        self.layer1 = self.d.itervalues()
        self.layer2 = None

    def __iter__(self):
        return self

    def __len__(self):
        i = 0
        for layer2 in self.d.itervalues():
            i += len(layer2)
        return i

    def next(self):
        try:
            return self.layer2.next()
        except (AttributeError, TypeError, StopIteration):
            self.layer2 = self.layer1.next().itervalues()
            return self.layer2.next()


class SkeletonIterator:
    # TODO: Something is not quite right here
    def __init__(self, skellike):
        if isinstance(skellike, Skeleton):
            self.skel = skellike
        else:
            self.skel = Skeleton(skellike)
        self.ptrs = deque([self.skel])
        self.l = None
        self.keyses = [self.skel.keys()]

    def __len__(self):
        if self.l is not None:
            return self.l
        i = 0
        h = SkeletonIterator(self.skel)
        while True:
            try:
                h.next()
                i += 1
            except StopIteration:
                self.l = i
                return i

    def __iter__(self):
        return self

    def next(self):
        while len(self.ptrs) > 0:
            try:
                ptr = self.ptrs.pop()
                keys = self.keyses.pop()
            except IndexError:
                raise StopIteration
            if ptr.rowdict:
                return ptr
            else:
                try:
                    k = keys.pop()
                except IndexError:
                    continue
                if len(keys) > 0:
                    self.ptrs.append(ptr)
                    self.keyses.append(keys)
                self.ptrs.append(ptr[k])
                self.keyses.append(ptr[k].keys())
        raise StopIteration


class ScissorOrderedGroup(pyglet.graphics.OrderedGroup):
    def __init__(self, order, parent, window,
                 left, top, bot, right, proportional=True):
        super(ScissorOrderedGroup, self).__init__(order, parent)
        self.window = window
        self.left = left
        self.top = top
        self.bot = bot
        self.right = right
        self.proportional = proportional

    def set_state(self):
        if self.proportional:
            l = int(self.left * self.window.width)
            b = int(self.bot * self.window.height)
            r = int(self.right * self.window.width)
            t = int(self.top * self.window.height)
        else:
            l = self.left
            b = self.bot
            r = self.right
            t = self.top
        w = r - l
        h = t - b
        pyglet.gl.glScissor(l, b, w, h)
        pyglet.gl.glEnable(pyglet.gl.GL_SCISSOR_TEST)

    def unset_state(self):
        pyglet.gl.glDisable(pyglet.gl.GL_SCISSOR_TEST)


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


class EmptyTabdict(Exception):
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

    def next(self):
        it = self.l_iter.next()
        i = self.i
        self.i += 1
        return (i, it)


class FilterIter:
    def __init__(self, itr, do_not_return):
        self.real = iter(itr)
        self.do_not_return = do_not_return

    def __iter__(self):
        return self

    def next(self):
        r = self.real.next()
        while r in self.do_not_return:
            r = self.real.next()
        return r


class FirstOfTupleFilter:
    def __init__(self, containable):
        self.containable = containable

    def __contains__(self, t):
        return t[0] in self.containable


class Timestream:
    __metaclass__ = SaveableMetaclass
    """A graph of many timelines, some of which share some of their time.

    The first argument is a dictionary keyed with branch indices, with
    values composed of tuples like:

    (parent_branch, start, end)

    parent_branch must be another key in the branchdict. start and end
    are the tick when a branch begins (which cannot change) and ends
    (which can, and probably will, perhaps as often as once per
    update).

    Call the update method to rearrange the contents of this board to
    reflect the state of the branches.

    """
    tables = [
        ("timestream",
         {"branch": "integer not null",
          "parent": "integer not null",
          "tick_from": "integer not null",
          "tick_to": "integer not null"},
         ("branch",),
         {"parent": ("timestream", "branch")},
         ["branch>=0", "tick_from>=0",
          "tick_to>=tick_from", "parent=0 or parent<>branch"])]

    def __init__(self, closet):
        self.closet = closet
        td = self.closet.skeleton
        self.branch_edges = defaultdict(set)
        self.branch_done_to = defaultdict(lambda: -1)
        self.branchdict = {}
        for rd in SkeletonIterator(td["timestream"]):
            self.branchdict[rd["branch"]] = (
                rd["parent"], rd["tick_from"], rd["tick_to"])
        self.graph = Graph(directed=True)
        self.graph.add_vertices(2)
        self.graph.vs["tick"] = [0, 0]
        self.graph.add_edge(0, 1, branch=0)
        self.branch_edges[0].add(0)
        self.branch_head = {0: self.graph.vs[0]}
        # When the player travels to the past and then branches the
        # timeline, it may result in a new vertex in the middle of
        # what once was an unbroken edge. The edge succeeding the new
        # vertex, representing how things went *originally*, is still
        # representative of the old branch, even though it is now a
        # successor of the vertex for a different branch
        # altogether. That original branch now has another edge
        # representing it.
        self.update_handlers = set()
        self.update(0)

    def __hash__(self):
        b = []
        for t in self.branchdict.itervalues():
            b.extend(t)
        return hash(tuple(b))


class Timestream:
    __metaclass__ = SaveableMetaclass
    """A graph of many timelines, some of which share some of their time.

    The first argument is a dictionary keyed with branch indices, with
    values composed of tuples like:

    (parent_branch, start, end)

    parent_branch must be another key in the branchdict. start and end
    are the tick when a branch begins (which cannot change) and ends
    (which can, and probably will, perhaps as often as once per
    update).

    Call the update method to rearrange the contents of this board to
    reflect the state of the branches.

    """
    tables = [
        ("timestream",
         {"branch": "integer not null",
          "parent": "integer not null",
          "tick_from": "integer not null",
          "tick_to": "integer not null"},
         ("branch",),
         {"parent": ("timestream", "branch")},
         ["branch>=0", "tick_from>=0",
          "tick_to>=tick_from", "parent=0 or parent<>branch"])]

    def __init__(self, closet):
        self.closet = closet
        td = self.closet.skeleton
        self.branch_edges = defaultdict(set)
        self.branch_done_to = defaultdict(lambda: -1)
        self.branchdict = td["timestream"]
        self.graph = Graph(directed=True)
        self.graph.add_vertices(2)
        self.graph.vs["tick"] = [0, 0]
        self.graph.add_edge(0, 1, branch=0)
        self.branch_edges[0].add(0)
        self.branch_head = {0: self.graph.vs[0]}
        # When the player travels to the past and then branches the
        # timeline, it may result in a new vertex in the middle of
        # what once was an unbroken edge. The edge succeeding the new
        # vertex, representing how things went *originally*, is still
        # representative of the old branch, even though it is now a
        # successor of the vertex for a different branch
        # altogether. That original branch now has another edge
        # representing it.
        self.update_handlers = set()
        self.update(0)

    def __hash__(self):
        b = []
        for t in self.branchdict.itervalues():
            b.extend(t)
        return hash(tuple(b))

    def update(self, ts=0):
        """Update the tree to reflect the current state of branchdict.

For every branch in branchdict, there should be one vertex at the
start and one at the end. If the branch has grown, but has as many
child branches as previously, change the tick of the end vertex to
reflect the growth.

If there are more child branches than before, split an edge to place a
vertex at the start point of each child branch. Then extend a new edge
out to the end of the child branch. The start and end vertices may
be on the same tick, in which case they are connected by an edge of
length zero.

        """
        logger.debug(
            "Updating timestream. Former "
            "branchdict: {0}".format(self.branchdict))
        for (branch, rd) in self.branchdict.iteritems():
            done_to = self.branch_done_to[branch]
            parent = rd["parent"]
            tick_from = rd["tick_from"]
            tick_to = rd["tick_to"]
            if tick_to > done_to:
                # I am now looking at a tick-window that has not been
                # put into the graph yet.
                #
                # Where does it belong?
                #
                # Is its branch, at least, already in the graph somewhere?
                if branch in self.branch_edges:
                    # I may have to extend an edge to make it fit the
                    # whole tick-window.
                    e_to = self.get_edge(branch, tick_to)
                    if e_to is None:
                        e_to = self.latest_edge(branch)
                        v = self.graph.vs[e_to.target]
                        growth = tick_to - v["tick"]
                        v["tick"] += growth
                    # Otherwise there's not really much to do here.
                elif branch not in self.branchdict:
                    # I assume that this dict reflects the genealogy
                    # of the branches accurately
                    try:
                        self.split_branch(
                            parent, branch, tick_from, tick_to - tick_from)
                    except KeyError:
                        assert(branch == 0)
                        self.graph.add_vertices(2)
                        self.graph.vs["tick"] = [0, tick_to]
                        self.graph.add_edge(0, 1, branch=0)
                        eid = self.graph.get_eid(0, 1)
                        self.branch_edges[0].add(eid)
                else:
                    self.extend_branch_to(branch, tick_to)
            self.branch_done_to[branch] = tick_to
            for handler in self.update_handlers:
                handler.on_timestream_update()
        logger.debug("Latter branchdict: {0}".format(self.branchdict))

    def get_edge_len(self, e):
        if isinstance(e, int):
            e = self.graph.es[e]
        vo = self.graph.vs[e.source]
        vd = self.graph.vs[e.target]
        return vd["tick"] - vo["tick"]

    def sanitize_vert(self, v):
        if isinstance(v, Vertex):
            vert = v
            vid = vert.index
        else:
            vid = v
            vert = self.graph.vs[vid]
        return (vert, vid)

    def sanitize_edge(self, e):
        if isinstance(e, Edge):
            edge = e
            eid = edge.index
        else:
            eid = e
            edge = self.graph.es[eid]
        return (edge, eid)

    def vertex_in_branch(self, v, branch):
        v = self.sanitize_vert(v)[1]
        for eid in self.graph.incident(v):
            if eid in self.branch_edges[branch]:
                return True
        return False

    def add_edge(self, branch, vert_from, vert_to):
        (vert_from, vi1) = self.sanitize_vert(vert_from)
        (vert_to, vi2) = self.sanitize_vert(vert_to)
        assert(branch in self.branchdict)
        assert(vert_from["tick"] <= vert_to["tick"])
        if branch not in self.branch_head:
            self.branch_head[branch] = vert_from
        self.graph.add_edge(vi1, vi2, branch=branch)
        eid = self.graph.get_eid(vi1, vi2)
        self.branch_edges[branch].add(eid)
        p = self.branchdict[branch]["parent"]
        if vert_from["tick"] < self.branchdict[branch]["tick_from"]:
            a = vert_from["tick"]
        else:
            a = self.branchdict[branch]["tick_from"]
        if vert_to["tick"] > self.branchdict[branch]["tick_to"]:
            z = vert_to["tick"]
        else:
            z = self.branchdict[branch]["tick_to"]
        self.branchdict[branch] = {
            "branch": branch,
            "parent": p,
            "tick_from": a,
            "tick_to": z}
        return self.graph.es[eid]

    def delete_edge(self, e):
        (e, eid) = self.sanitize_edge(e)
        old_branch = e["branch"]
        self.branch_edges[old_branch].discard(eid)
        self.graph.delete_edges(eid)

    def add_vert(self, tick):
        i = len(self.graph.vs)
        self.graph.add_vertex(tick=tick)
        v = self.graph.vs[i]
        return v

    def vert_branch(self, vert):
        if isinstance(vert, int):
            vert = self.graph.vs[vert]
        try:
            eid = self.graph.incident(vert)[0]
            return self.graph.es[eid]["branch"]
        except:
            return -1

    def add_vert_on(self, e, tick):
        (e, eid) = self.sanitize_edge(e)
        former = self.graph.vs[e.source]
        latter = self.graph.vs[e.target]
        old_branch = e["branch"]
        v = self.add_vert(tick)
        i = v.index
        e1 = self.add_edge(old_branch, former, i)
        e2 = self.add_edge(old_branch, i, latter)
        return (e1, v, e2)

    def get_edge(
            self,
            vert_from_or_branch,
            vert_to_or_tick,
            mode="branch_tick"):
        if mode == "branch_tick":
            return self.get_edge_from_branch_tick(
                vert_from_or_branch,
                vert_to_or_tick)
        elif mode == "verts":
            return self.get_edge_from_verts(
                vert_from_or_branch,
                vert_to_or_tick)
        else:
            raise Exception("Invalid mode")

    def get_edge_from_verts(self, vert_from, vert_to):
        if isinstance(vert_from, Vertex):
            vert_from = vert_from.index
        if isinstance(vert_to, Vertex):
            vert_to = vert_to.index
        eid = self.graph.get_eid(vert_from, vert_to)
        return self.graph.es[eid]

    def get_edge_from_branch_tick(self, branch, tick):
        """Return the edge that contains the given tick in the given branch,
or None if no such edge exists."""
        v = self.branch_head[branch]
        if tick < v["tick"]:
            raise Exception("This branch started after that tick.")
        return self.successor_on_branch_tick(v, branch, tick)

    def successor_on_branch_tick(self, v, branch, tick):
        """Traverse the graph starting from the given vertex. Return the edge
containing the given tick in the given branch. If it doesn't exist,
return None."""
        if tick == v["tick"]:
            # I'll consider ticks coinciding exactly with a vertex to
            # be in the descendant edge in that branch.
            if self.vertex_in_branch(v, branch):
                for e in self.graph.incident(v):
                    if e in self.branch_edges[branch]:
                        return e
            return None
        for eid in self.graph.incident(v):
            e = self.graph.es[eid]
            v_to = self.graph.vs[e.target]
            if eid in self.branch_edges[branch]:
                if v_to["tick"] > tick:
                    return e
                else:
                    # None of these edges are right! You want the ones
                    # after this vertex here.
                    return self.successor_on_branch_tick(v_to, branch, tick)
            elif v_to["tick"] >= tick:
                return e
        return None

    def split_branch(self, old_branch, new_branch, tick_from, tick_to):
        """Find the edge in old_branch in the given tick, split it, and start
a new edge off the split. The new edge will be a member of
new_branch.

        """
        assert(new_branch not in self.branchdict)
        self.branchdict[new_branch] = {
            "parent": old_branch,
            "tick_from": tick_from,
            "tick_to": tick_to}
        e = self.get_edge_from_branch_tick(old_branch, tick_from)
        if e is None:
            vseq = self.graph.vs(tick_eq=tick_from)
            v1 = vseq[0]
            v2 = self.add_vert(tick=tick_to)
        else:
            (e1, v1, e2) = self.add_vert_on(e, tick_from)
            v2 = self.add_vert(tick=tick_to)
        return self.add_edge(new_branch, v1, v2)

    def latest_edge(self, branch):
        """Return the edge in the given branch that ends on the highest
tick."""
        edges = set(self.branch_edges[branch])
        late = self.graph.es[edges.pop()]
        v_late = self.graph.vs[late.target]
        while len(edges) > 0:
            e = self.graph.es[edges.pop()]
            v_e = self.graph.vs[e.target]
            if v_e["tick"] > v_late["tick"]:
                late = e
                v_late = v_e
        return late

    def extend_branch(self, branch, n):
        """Make the branch so many ticks longer."""
        edge = self.latest_edge(branch)
        vert = self.graph.vs[edge.target]
        vert["tick"] += n
        self.branchdict[branch]["tick_to"] += n

    def extend_branch_to(self, branch, tick_to):
        """Make the branch end on the given tick, but only if it is later than
the branch's current end."""
        try:
            edge = self.latest_edge(branch)
        except KeyError:
            v1 = self.add_vert(0)
            v2 = self.add_vert(tick_to)
            return self.add_edge(branch, v1, v2)
        vert = self.graph.vs[edge.target]
        if tick_to > vert["tick"]:
            vert["tick"] = tick_to
        if tick_to > self.branchdict[branch]["tick_to"]:
            self.branchdict[branch]["tick_to"] = tick_to


class FakeCloset:
    def __init__(self, skellike):
        self.skeleton = Skeleton(skellike)
        self.timestream = Timestream(self)


class BranchError(Exception):
    pass
