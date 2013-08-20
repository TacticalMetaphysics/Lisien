# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import pyglet
import ctypes
from math import sqrt, hypot, atan, pi, sin, cos
from logging import getLogger

logger = getLogger(__name__)

phi = (1.0 + sqrt(5))/2.0

schemata = []

colnames = {}

colnamestr = {}

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

To save, you need to define a method called get_tabdict. It should
return a dictionary where the keys are table names. The values are
either rowdicts or iterables over rowdicts. A rowdict is a dictionary
containing the information in a single record of a table; the keys are
the names of the fields.

To load, you need to define a method called from_tabdict that takes
that same kind of dictionary and returns an instance of your class.

Once you've defined those, the save(db) and load(db) methods will save
or load your class in the given database. If you need to create the
database, look at the schemata attribute: execute that in a SQL cursor
and your table will be ready.

    """
    def __new__(metaclass, clas, parents, attrs):
        tablenames = []
        primarykeys = {}
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
            prelude = None
        for tabtup in tablist:
            (name, decls, pkey, fkeys, cks) = tabtup
            tablenames.append(name)
            coldecls[name] = decls
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
        for item in primarykeys.iteritems():
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
            fkeystrs = ["FOREIGN KEY (%s) REFERENCES %s(%s)" %
                        (item[0], item[1][0], item[1][1])
                        for item in fkeys.iteritems()]
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
            schemata.append((
                tablename, set([
                    fkey[0] for fkey in
                    fkeys.itervalues()]), create_stmt, prelude))

        def gen_sql_insert(rowdicts, tabname):
            if rowdicts == []:
                return []
            sample = rowdicts[0]
            cols_used = [col for col in colnames[tabname] if col in sample]
            colsstr = ", ".join(cols_used)
            row_qms = ", ".join(["?"] * len(sample))
            rowstr = "({0})".format(row_qms)
            rowsstr = ", ".join([rowstr] * len(rowdicts))
            qrystr = inserts[tabname].format(colsstr, rowsstr)
            qrylst = []
            for rowdict in iter(rowdicts):
                for col in cols_used:
                    qrylst.append(rowdict[col])
            qrytup = tuple(qrylst)
            return (qrystr, qrytup)

        def insert_rowdicts_table(db, rowdicts, tabname):
            db.c.execute(*gen_sql_insert(rowdicts, tabname))
            return []

        def gen_sql_delete(keydicts, tabname):
            keyns = keynames[tabname]
            keys = []
            wheres = []
            for keydict in iter(keydicts):
                checks = []
                for keyn in keyns:
                    checks.append(keyn + "=?")
                    keys.append(keydict[keyn])
                wheres.append("(" + " AND ".join(checks) + ")")
            wherestr = " OR ".join(wheres)
            qrystr = "DELETE FROM {0} WHERE {1}".format(tabname, wherestr)
            return (qrystr, tuple(keys))

        def delete_keydicts_table(db, keydicts, tabname):
            db.c.execute(*gen_sql_delete(keydicts, tabname))

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

        def detect_keydicts_table(db, keydicts, tabname):
            db.c.execute(*gen_sql_detect(keydicts, tabname))
            return db.c.fetchall()

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

        def missing_keydicts_table(db, keydicts, tabname):
            db.c.execute(*gen_sql_missing(keydicts, tabname))
            return db.c.fetchall()

        def insert_tabdict(db, tabdict):
            for item in tabdict.iteritems():
                (tabname, rd) = item
                insert_rowdicts_table(db, rd, tabname)

        def delete_tabdict(db, tabdict):
            qryfmt = "DELETE FROM {0} WHERE {1}"
            for (tabn, rows) in tabdict.iteritems():
                if rows == []:
                    continue
                vals = []
                ors = []
                for row in iter(rows):
                    keyns = keynames[tabn]
                    ands = []
                    for keyn in keyns:
                        ands.append(keyn + "=?")
                        vals.append(row[keyn])
                    ors.append("(" + " AND ".join(ands) + ")")
                qrystr = qryfmt.format(tabn, " OR ".join(ors))
                qrytup = tuple(vals)
                db.c.execute(qrystr, qrytup)

        def detect_tabdict(db, tabdict):
            r = {}
            for item in tabdict.iteritems():
                (tabname, rd) = item
                if isinstance(rd, dict):
                    r[tabname] = detect_keydicts_table(db, [rd], tabname)
                else:
                    r[tabname] = detect_keydicts_table(db, rd, tabname)
            return r

        def missing_tabdict(db, tabdict):
            r = {}
            for item in tabdict.iteritems():
                (tabname, rd) = item
                if isinstance(rd, dict):
                    r[tabname] = missing_keydicts_table(db, [rd], tabname)
                else:
                    r[tabname] = missing_keydicts_table(db, rd, tabname)
            return r

        def coresave(self):
            td = self.get_tabdict()
            logger.debug("writing a tabdict to disk")
            for item in td.iteritems():
                logger.debug("--in the table %s:", item[0])
                i = 0
                for record in item[1]:
                    logger.debug("----row %d:", i)
                    i += 1
                    for (key, val) in record.iteritems():
                        logger.debug("------%s = %s", key, val)
            delete_tabdict(self.rumor, td)
            insert_tabdict(self.rumor, td)

        def save(self):
            coresave(self)

        def get_keydict(self):
            tabd = self.get_tabdict()
            r = {}
            for tabn in tablenames:
                r[tabn] = {}
                for keyn in keynames[tabn]:
                    r[tabn][keyn] = tabd[tabn][keyn]
            return r

        def erase(self):
            delete_tabdict(self.rumor, self.get_keydict())

        atrdic = {
            'insert_tabdict': lambda self, db, td: insert_tabdict(db, td),
            'delete_tabdict': lambda self, db, td: delete_tabdict(db, td),
            'detect_tabdict': lambda self, db, td: detect_tabdict(db, td),
            'missing_tabdict': lambda self, db, td: missing_tabdict(db, td),
            'gen_sql_insert': lambda self, rd, tn: gen_sql_insert(rd, tn),
            'gen_sql_delete': lambda self, rd, tn: gen_sql_delete(rd, tn),
            'gen_sql_detect': lambda self, rd, tn: gen_sql_detect(rd, tn),
            'gen_sql_missing': lambda self, rd, tn: gen_sql_missing(rd, tn),
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
            'coresave': coresave,
            'save': save,
            'maintab': tablenames[0],
            'get_keydict': get_keydict,
            'erase': erase}
        atrdic.update(attrs)

        return type.__new__(metaclass, clas, parents, atrdic)


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


def compile_tabdicts(objs):
    tabdicts = [o.tabdict for o in objs]
    mastertab = {}
    for tabdict in tabdicts:
        for item in tabdict.iteritems():
            (tabname, rowdict) = item
            if tabname not in mastertab:
                mastertab[tabname] = []
            mastertab[tabname].append(rowdict)
    return mastertab


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
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.imagery:
            return None
        for (tick_from, (img, tick_to)) in self.imagery[branch].iteritems():
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
                assert(hasattr(img, 'tex'))
                return img
        return None

    def set_img(self, img, branch=None, tick_from=None, tick_to=None):
        assert(hasattr(img, 'tex'))
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.imagery:
            self.imagery[branch] = {}
        if branch in self.indefinite_imagery:
            (indef_img, indef_start) = self.indefinite_imagery[branch]
            if tick_to is None:
                del self.imagery[branch][indef_start]
                self.imagery[branch][tick_from] = (img, None)
                self.indefinite_imagery[branch] = (img, tick_from)
            else:
                if tick_from < indef_start:
                    if tick_to < indef_start:
                        self.imagery[branch][tick_from] = (img, tick_to)
                    elif tick_to == indef_start:
                        del self.indefinite_imagery[branch]
                        self.imagery[branch][tick_from] = (img, tick_to)
                    else:
                        del self.imagery[branch][indef_start]
                        del self.indefinite_imagery[branch]
                        self.imagery[branch][tick_from] = (img, tick_to)
                elif tick_from == indef_start:
                    del self.indefinite_imagery[branch]
                    self.imagery[branch][tick_from] = (img, tick_to)
                else:
                    self.imagery[branch][indef_start] = (
                        indef_img, tick_from-1)
                    del self.indefinite_imagery[branch]
                    self.imagery[branch][tick_from] = (img, tick_to)
        else:
            self.imagery[branch][tick_from] = (img, tick_to)
            if tick_to is None:
                self.indefinite_imagery[branch] = (img, tick_from)

    def new_branch_imagery(self, parent, branch, tick):
        for (tick_from, (img, tick_to)) in self.imagery[parent].iteritems():
            if tick_to >= tick or tick_to is None:
                if tick_from < tick:
                    self.imagery[branch][tick] = (img, tick_to)
                    if tick_to is None:
                        self.indefinite_imagery[branch] = tick
                else:
                    self.imagery[branch][tick_from] = (img, tick_to)
                    if tick_to is None:
                        self.indefinite_imagery[branch] = tick_from


class TerminableInteractivity:
    __metaclass__ = SaveableMetaclass

    def is_interactive(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.interactivity:
            return False
        for (tick_from, tick_to) in self.interactivity[branch].iteritems():
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
                return True
        return False

    def set_interactive(self, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.interactivity:
            self.interactivity[branch] = {}
        if branch in self.indefinite_interactivity:
            prevstart = self.indefinite_interactivity[branch]
            if tick_to is None:
                # Two indefinite periods of interactivity cannot coexist.
                # Assume that you meant to overwrite the old one.
                del self.interactivity[branch][prevstart]
                self.indefinite_interactivity[branch] = tick_from
                self.interactivity[branch][tick_from] = None
            else:
                if tick_from < prevstart:
                    if tick_to > prevstart:
                        # You had an indefinite period of interactivity,
                        # and asked to overwrite a span of it--from the
                        # beginning to some tick--with part of a definite
                        # period of interactivity.
                        #
                        # That's a bit weird. The only way to really
                        # comply with that request is to delete the
                        # indefinite period.
                        del self.interactivity[branch][prevstart]
                        del self.indefinite_interactivity[branch]
                        self.interactivity[branch][tick_from] = tick_to
                    elif tick_to == prevstart:
                        # Putting a definite period of interactivity
                        # on before the beginning of an indefinite one
                        # is equivalent to rescheduling the start of
                        # the indefinite one.
                        del self.interactivity[branch][prevstart]
                        self.interactivity[branch][tick_from] = None
                    else:
                        # This case I can simply schedule like normal.
                        self.interactivity[branch][tick_from] = tick_to
                elif tick_from == prevstart:
                    # Assume you mean to overwrite
                    self.interactivity[branch][tick_from] = tick_to
                    del self.indefinite_interactivity[branch]
                else:
                    # By scheduling the start of something definite
                    # after the start of something indefinite, you've
                    # implied that the indefinite thing shouldn't be
                    # so indefinite after all.
                    self.interactivity[branch][prevstart] = tick_from - 1
                    del self.indefinite_interactivity[branch]
        else:
            self.interactivity[branch][tick_from] = tick_to
            if tick_to is None:
                self.indefinite_interactivity[branch] = tick_from

    def new_branch_interactivity(self, parent, branch, tick):
        for (tick_from, tick_to) in self.interactivity[parent].iteritems():
            if tick_to >= tick or tick_to is None:
                if tick_from < tick:
                    self.interactivity[branch][tick] = tick_to
                    if tick_to is None:
                        self.indefinite_interactivity[branch] = tick
                else:
                    self.interactivity[branch][tick_from] = tick_to
                    if tick_to is None:
                        self.indefinite_interactivity[branch] = tick_from


class TerminableCoords:
    __metaclass__ = SaveableMetaclass

    def get_coords(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.coord_dict:
            return None
        it = self.coord_dict[branch].iteritems()
        for (tick_from, (x, y, tick_to)) in it:
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
                return (x, y)
        return None

    def set_coords(self, x, y, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.coord_dict:
            self.coord_dict[branch] = {}
        if str(self) == 'myroom':
            pass
        if branch in self.indefinite_coords:
            (ix, iy, itf) = self.indefinite_coords[branch]
            if tick_to is None:
                self.coord_dict[branch][itf] = (ix, iy, tick_from - 1)
                self.coord_dict[branch][tick_from] = (x, y, tick_to)
            else:
                if tick_from < itf:
                    if tick_to < itf:
                        self.coord_dict[branch][tick_from] = (x, y, tick_to)
                    elif tick_to == itf:
                        self.coord_dict[branch][tick_from] = (x, y, None)
                        del self.coord_dict[branch][itf]
                    else:
                        del self.indefinite_coords[branch]
                        del self.coord_dict[branch][itf]
                        self.coord_dict[branch][tick_from] = (x, y, tick_to)
                elif tick_from == itf:
                    del self.indefinite_coords[branch]
                    self.coord_dict[branch][tick_from] = (x, y, tick_to)
                else:
                    self.coord_dict[branch][itf] = (ix, iy, tick_from - 1)
                    del self.indefinite_coords[branch]
                    self.coord_dict[branch][tick_from] = (x, y, tick_to)
        else:
            self.coord_dict[branch][tick_from] = (x, y, tick_to)
        if tick_to is None:
            self.indefinite_coords[branch] = (x, y, tick_from)

    def new_branch_coords(self, parent, branch, tick):
        for (tick_from, (x, y, tick_to)) in (
                self.coord_dict[parent].iteritems()):
            if tick_to >= tick or tick_to is None:
                if tick_from < tick:
                    self.coord_dict[branch][tick] = (x, y, tick_to)
                    if tick_to is None:
                        self.indefinite_coords[branch] = tick
                else:
                    self.coord_dict[branch][tick_from] = (x, y, tick_to)
                    if tick_to is None:
                        self.indefinite_coords[branch] = tick_from


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
