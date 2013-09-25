# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
import pyglet
import ctypes
from math import sqrt, hypot, atan, pi, sin, cos
from logging import getLogger
from sqlite3 import IntegrityError
from collections import deque, MutableMapping

logger = getLogger(__name__)

phi = (1.0 + sqrt(5))/2.0

schemata = {}

colnames = {}

colnamestr = {}

primarykeys = {}

tabclas = {}

saveables = []

saveable_classes = []


class SkelRowIter(object):
    def __init__(self, skel):
        self.skel = skel
        self.ptrs = deque([self.skel])
        self.keyses = [self.skel.keys()]

    def __len__(self):
        if hasattr(self, 'l'):
            return self.l
        i = 0
        h = SkelRowIter(self.skel)
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


class Skeleton(MutableMapping):
    def __init__(self, content, name="", parent=None,
                 set_listener=None, del_listener=None):
        self.rowdict = None
        self.name = name
        self.parent = parent
        self.set_listener = set_listener
        self.del_listener = del_listener
        if isinstance(content, Skeleton):
            content = content.content
        self.content = {}
        for (k, v) in content.iteritems():
            if self.rowdict is None:
                self.rowdict = v.__class__ in (str, int, float, type(None))
            elif self.rowdict is True:
                assert(v.__class__ in (str, int, float, type(None)))
            else:
                assert(v.__class__ in (dict, Skeleton))
            self[k] = v

    def __getitem__(self, k):
        return self.content[k]

    def __setitem__(self, k, v):
        assert(not isinstance(v, list))
        if self.rowdict:
            self.content[k] = v
        else:
            self.content[k] = Skeleton(v, k, self)
        if self.set_listener is not None:
            self.listener((str(self),), k, v)
        if self.parent is not None:
            self.parent.on_child_set((str(self),), k, v)

    def __delitem__(self, k):
        todel = self.content[k]
        del self.content[k]
        if self.del_listener is not None:
            self.listener((str(self),), k, todel)
        if self.parent is not None:
            self.parent.on_child_delete((str(self),), k, todel)

    def __iter__(self):
        return iter(self.content)

    def __len__(self):
        return len(self.content)

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
        if self.rowdict and self.content == other.content:
            self.content = {}
            return
        for (k, v) in other.iteritems():
            if k not in self:
                continue
            elif v == self[k]:
                del self[k]
            elif self[k].rowdict:
                continue
            else:
                self[k] -= v
        return self

    def __sub__(self, other):
        selfie = self.copy()
        selfie -= other
        return selfie

    def __str__(self):
        return str(self.name)

    def keys(self):
        return sorted(self.content.keys())

    def key_before(self, k):
        anterior = [j for j in self.content.iterkeys() if j < k]
        if anterior == []:
            raise KeyError("There is nothing before {0}".format(k))
        return max(anterior)

    def key_after(self, k):
        posterior = [j for j in self.content.iterkeys() if j > k]
        if posterior == []:
            raise KeyError("There is nothing after {0}".format(k))
        return min(posterior)

    def copy(self):
        if self.rowdict:
            return Skeleton(dict(self.content))
        else:
            r = {}
            for (k, v) in self.content.iteritems():
                r[k] = v.copy()
            return Skeleton(r)

    def deepcopy(self):
        if self.rowdict:
            return Skeleton(self.content, self.name,
                            self.parent, self.listeners)
        else:
            r = {}
            for (k, v) in self.content.iteritems():
                r[k] = v.deepcopy()
            return Skeleton(r, self.name, self.parent,
                            self.listeners)

    def update(self, skellike):
        for (k, v) in skellike.iteritems():
            if v.rowdict:
                self[k] = v
                continue
            v = Skeleton(v, k, self)
            if k in self.content:
                self.content[k].update(v)
            else:
                self.content[k] = v

    def iterrows(self):
        return SkelRowIter(self)

    def on_child_set(self, childn, k, v):
        qn = (str(self),) + childn
        if self.set_listener is not None:
            self.set_listener(qn, k, v)
        if self.parent is not None:
            self.parent.on_child_set(qn, k, v)

    def on_child_delete(self, childn, k, v):
        qn = (str(self),) + childn
        if self.del_listener is not None:
            self.del_listener(qn, k, v)
        if self.parent is not None:
            self.parent.on_child_delete(qn, k, v)


def empty_skel():
    tns = []
    for saveable in saveables:
        tns.extend(saveable[3])
    r = Skeleton({})
    for tn in tns:
        r[tn] = {}
    return r


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
                itr = Skeleton(rowdicts[tabname]).iterrows()
            else:
                itr = Skeleton(rowdicts).iterrows()
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
            except EmptyTabdict:
                return

        def gen_sql_delete(keydicts, tabname):
            try:
                keyns = keynames[tabname]
            except KeyError:
                return
            keys = []
            wheres = []
            kitr = Skeleton(keydicts).iterrows()
            if len(kitr) == 0 or tabname not in tablenames:
                raise EmptyTabdict
            for keydict in kitr:
                checks = []
                for keyn in keyns:
                    checks.append(keyn + "=?")
                    keys.append(keydict[keyn])
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
            kitr = Skeleton(keydicts).iterrows()
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
            kitr = Skeleton(keydicts).iterrows()
            for keydict in kitr:
                for key in keys:
                    if key in keydict:
                        qrylst.append(keydict[key])
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
                rd = dictify_row(row, colnames[tabname])
                ptr = r
                lptr = r
                for key in primarykeys[tabname]:
                    if rd[key] not in ptr:
                        ptr[rd[key]] = {}
                    lptr = ptr
                    ptr = ptr[rd[key]]
                lptr[rd[key]] = rd
            return Skeleton({tabname: r})

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
        if branch not in self.imagery:
            return None
        prev = None
        for rd in self.imagery[branch].iterrows():
            if rd["tick_from"] > tick:
                break
            else:
                prev = rd
        if prev is None or prev["img"] in ("", None):
            return None
        else:
            return self.closet.get_img(prev["img"])

    def new_branch_imagery(self, parent, branch, tick):
        prev = None
        started = False
        for tick_from in self.imagery[parent]:
            if tick_from >= tick:
                rd2 = dict(self.imagery[parent][tick_from])
                rd2["branch"] = branch
                if branch not in self.imagery:
                    self.imagery[branch] = {}
                self.imagery[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(self.imagery[parent][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    self.imagery[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from


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
        for rd in self.interactivity.iterrows():
            if rd["tick_from"] <= tick and tick <= rd["tick_to"]:
                return True
        return False

    def new_branch_interactivity(self, parent, branch, tick):
        prev = None
        started = False
        for tick_from in self.interactivity[parent]:
            if tick_from >= tick:
                rd2 = dict(self.interactivity[parent][tick_from])
                rd2["branch"] = branch
                if branch not in self.interactivity:
                    self.interactivity[branch] = {}
                self.interactivity[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(self.interactivity[parent][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    self.interactivity[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from


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
        return litter[:-1] + [litter[-2].next().itervalues()]
    except StopIteration:
        if len(litter) <= 1:
            raise StopIteration
        nvi = next_val_iter(litter[:-1])
        return nvi + [nvi[-1].next().itervalues()]


def skel_nth_generator(skel, n):
    iters = [skel.itervalues()]
    for i in xrange(0, n-1):
        iters.append(iters[-1].next().itervalues())
    try:
        yield iters[-1].next()
    except StopIteration:
        if len(iters) <= 1:
            raise StopIteration
        iters = next_val_iter(iters)
        yield iters[-1].next()


class Timestream(object):
    # I think updating the start and end ticks of a branch using
    # listeners might be a good idea
    __metaclass__ = SaveableMetaclass

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
        self.skeleton = self.closet.skeleton
        self.hi_branch = 0
        for tab in self.listen_tables:
            self.skeleton[tab].set_listener = self.skel_set

    def skel_set(self, qn, k, v):
        if k == "branch":
            if v > self.hi_branch:
                self.hi_branch = v

    def branches(self, table=None):
        if table is None:
            return self.allbranches()
        else:
            return self.branchtable(table)

    def branchtable(self, table):
        n = self.tab_depth[table]
        if n == 1:
            for d in self.skeleton[table].itervalues():
                for k in d.iterkeys():
                    yield k
        else:
            for d in skel_nth_generator(self.skeleton[table], n):
                for k in d.iterkeys():
                    yield k

    def allbranches(self):
        for (tabn, n) in self.tab_depth.iteritems():
            if n == 1:
                for d in self.skeleton[tabn].itervalues():
                    for k in d.iterkeys():
                        yield k
            else:
                for d in skel_nth_generator(self.skeleton[tabn], n):
                    for k in d.iterkeys():
                        yield k

    def allticks(self):
        for (tabn, n) in self.tab_depth.iteritems():
                if n == 1:
                    for d in self.skeleton[tabn].itervalues():
                        for k in d.iterkeys():
                            yield k
                else:
                    for d in skel_nth_generator(self.skeleton[tabn], n):
                        for k in d.iterkeys():
                            yield k

    def branchticks(self, branch):
        for (tabn, n) in self.tab_depth.iteritems():
            if n == 1:
                for d in self.skeleton[tabn].itervalues():
                    for k in d[branch].iterkeys():
                        yield k
            else:
                n = self.tab_depth[tabn]
                ptr = self.skeleton[tabn]
                try:
                    for i in xrange(0, n):
                        ptr = ptr.itervalues().next()
                except StopIteration:
                    continue
                for k in ptr[branch].iterkeys():
                    yield k

    def tabticks(self, table):
        n = self.tab_depth[table]
        if n == 1:
            for d in self.skeleton[table]:
                for k in d.iterkeys():
                    yield k
        else:
            for d in skel_nth_generator(self.skeleton[table], n):
                for k in d.iterkeys():
                    yield k

    def branchtabticks(self, branch, table):
        n = self.tab_depth[table]
        if n == 1:
            for k in self.skeleton[table][branch].iterkeys():
                yield k
        else:
            ptr = self.skeleton[table]
            for i in xrange(0, n):
                ptr = ptr.itervalues().next()
            if branch in ptr:
                for k in ptr[branch].iterkeys():
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

    def parent(self, branch):
        assert(branch > 0)
        return self.skeleton["timestream"][branch]["parent"]

    def children(self, branch):
        for rd in self.skeleton["timestream"].iterrows():
            if rd["parent"] == branch:
                yield rd["branch"]
