# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from math import sqrt, hypot, atan, pi, sin, cos
from logging import getLogger
from sqlite3 import IntegrityError
from collections import deque, MutableMapping
from kivy.uix.widget import WidgetMetaclass
from kivy.properties import NumericProperty
from kivy.event import EventDispatcher

logger = getLogger(__name__)

phi = (1.0 + sqrt(5))/2.0

schemata = {}

colnames = {}

colnamestr = {}

primarykeys = {}

tabclas = {}

saveables = []

saveable_classes = []


class Touchy(object):
    def on_touch_move(self, touch):
        if self.dragging:
            if not self.collide_point(touch.x, touch.y):
                self.dragging = False

    def on_touch_up(self, touch):
        self.dragging = False


def get_rd_during(skel, branch, tick):
    if branch not in skel:
        return None
    prev = None
    for tick_from in skel[branch]:
        if tick_from == tick:
            return skel[branch][tick_from]
        elif tick_from > tick:
            break
        else:
            prev = skel[branch][tick_from]
    return prev


class SkelRowIter(object):
    def __init__(self, skel):
        self.skel = skel
        self.ptrs = deque([self.skel])
        self.keyses = [list(self.skel.keys())]

    def __len__(self):
        if hasattr(self, 'l'):
            return self.l
        i = 0
        h = SkelRowIter(self.skel)
        while True:
            try:
                next(h)
                i += 1
            except StopIteration:
                self.l = i
                return i

    def __iter__(self):
        return self

    def __next__(self):
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
                    k = keys.pop(0)
                except IndexError:
                    continue
                if len(keys) > 0:
                    self.ptrs.append(ptr)
                    self.keyses.append(keys)
                self.ptrs.append(ptr[k])
                self.keyses.append(list(ptr[k].keys()))
        raise StopIteration

    def next(self):
        return self.__next__()


class Skeleton(MutableMapping, EventDispatcher):
    touches = NumericProperty(0)

    def __init__(self, content, name="", parent=None,
                 set_listener=None, del_listener=None):
        self.rowdict = None
        self.name = name
        self.parent = parent
        self.set_listener = set_listener
        self.del_listener = del_listener
        if hasattr(content, 'content'):
            content = content.content
        self.content = {}
        if isinstance(content, dict):
            kitr = content.iteritems()
        else:
            kitr = ListItemIterator(content)
        for (k, v) in kitr:
            if self.rowdict is None and v is not None:
                self.rowdict = v.__class__ in (str, unicode, int, float)
            elif self.rowdict is True:
                assert(v.__class__ in (str, unicode, int, float, type(None)))
            elif v is None:
                continue
            else:
                assert(v.__class__ in (dict, list, Skeleton))
            self[k] = v

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
        def really_set(k, v):
            if len(self.content) == 0:
                if isinstance(k, int):
                    self.ikeys = set()
                    self.content = []
                else:
                    self.content = {}
            if isinstance(k, int):
                assert(isinstance(self.content, list))
                while len(self.content) <= k:
                    self.content.append(None)
                self.content[k] = v
                self.ikeys.add(k)
            else:
                assert(isinstance(self.content, dict))
                self.content[k] = v

        if self.rowdict:
            really_set(k, v)
        else:
            really_set(k, Skeleton(v, k, self))
        self.touches += 1
        if self.parent is not None:
            self.parent.on_child_set((str(self),), k, v)

    def __delitem__(self, k):
        todel = self.content[k]
        if isinstance(self.content, dict):
            del self.content[k]
        else:
            self.ikeys.remove(k)
        self.touches += 1
        if self.parent is not None:
            self.parent.on_child_delete((str(self),), k, todel)

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
        if self.rowdict and self.content == other.content:
            self.content = {}
            return
        for (k, v) in other.items():
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
        if isinstance(self.content, dict):
            return list(self.content.keys())
        else:
            return sorted(self.ikeys)

    def key_before(self, k):
        anterior = [j for j in self.ikeys if j < k]
        if anterior == []:
            raise KeyError("There is nothing before {0}".format(k))
        return max(anterior)

    def key_after(self, k):
        posterior = [j for j in self.ikeys if j > k]
        if posterior == []:
            raise KeyError("There is nothing after {0}".format(k))
        return min(posterior)

    def copy(self):
        if self.rowdict:
            return Skeleton(dict(self.content))
        elif isinstance(self.content, list):
            r = {}
            for k in self.ikeys:
                r[k] = self.content[k].copy()
            return Skeleton(r)
        else:
            r = {}
            for (k, v) in self.content.items():
                r[k] = v.copy()
            return Skeleton(r)

    def deepcopy(self):
        if self.rowdict:
            return Skeleton(self.content, self.name,
                            self.parent, self.listeners)
        else:
            r = {}
            for (k, v) in self.content.items():
                r[k] = v.deepcopy()
            return Skeleton(r, self.name, self.parent,
                            self.listeners)

    def update(self, skellike):
        for (k, v) in skellike.items():
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
        self.touches += 1
        if self.parent is not None:
            self.parent.on_child_set(qn, k, v)

    def on_child_delete(self, childn, k, v):
        qn = (str(self),) + childn
        self.touches += 1
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
        for item in local_pkeys.items():
            (tablename, pkey) = item
            keynames[tablename] = sorted(pkey)
            keylen[tablename] = len(pkey)
            keyqms[tablename] = ", ".join(["?"] * keylen[tablename])
            keystrs[tablename] = "(" + keyqms[tablename] + ")"
        for item in coldecls.items():
            (tablename, coldict) = item
            valnames[tablename] = sorted(
                [key for key in list(coldict.keys())
                 if key not in keynames[tablename]])
            rowlen[tablename] = len(coldict)
            rowqms[tablename] = ", ".join(["?"] * rowlen[tablename])
            rowstrs[tablename] = "(" + rowqms[tablename] + ")"
        for tablename in coldecls.keys():
            colnames[tablename] = keynames[tablename] + valnames[tablename]
        for tablename in tablenames:
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
            (demands, provides, prelude, tablenames, postlude))

        def gen_sql_insert(rowdicts, tabname):
            if tabname in rowdicts:
                itr = Skeleton(rowdicts[tabname]).iterrows()
            else:
                itr = Skeleton(rowdicts).iterrows()
            if len(itr) == 0 or tabname not in tablenames:
                raise EmptySkeleton
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
                print(ie)
                print(gen_sql_insert(rowdicts, tabname))
            except EmptySkeleton:
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
                raise EmptySkeleton
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
            except EmptySkeleton:
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
            for (tabname, rdd) in td.items():
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
            for (tabname, rds) in skeleton.items():
                if tabname in tablenames:
                    insert_rowdicts_table(c, rds, tabname)

        @staticmethod
        def _delete_skeleton(c, skeleton):
            for (tabname, rds) in skeleton.items():
                if tabname in tablenames:
                    delete_keydicts_table(c, rds, tabname)

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


class SaveableWidgetMetaclass(WidgetMetaclass, SaveableMetaclass):
    pass


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


class Timestream(EventDispatcher):
    __metaclass__ = SaveableMetaclass
    # I think updating the start and end ticks of a branch using
    # listeners might be a good idea
    hi_branch = NumericProperty(0)
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
        try:
            self.hi_branch = max(self.branches())
        except ValueError:
            self.hi_branch = 0
        try:
            self.hi_tick = max(self.ticks())
        except ValueError:
            self.hi_tick = 0
        for tab in self.listen_tables:
            self.closet.skeleton[tab].set_listener = self.skel_set

    def skel_set(self, qn, k, v):
        pass

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

    def parent(self, branch):
        assert(branch > 0)
        return self.closet.skeleton["timestream"][branch]["parent"]

    def children(self, branch):
        for rd in self.closet.skeleton["timestream"].iterrows():
            if rd["parent"] == branch:
                yield rd["branch"]


class EmptySkeleton(Exception):
    pass
