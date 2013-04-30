from util import SaveableMetaclass, dictify_row
from style import Color, Style
from menu import Menu, MenuItem
import dimension
import color


__metaclass__ = SaveableMetaclass


class Board:
    tablenames = ["board", "board_menu"]
    coldecls = {"board":
                {"dimension": "text",
                 "width": "integer",
                 "height": "integer",
                 "wallpaper": "text"},
                "board_menu":
                {"board": "text",
                 "menu": "text"}}
    primarykeys = {"board": ("dimension",),
                   "board_menu": tuple()}
    foreignkeys = {"board":
                   {"dimension": ("dimension", "name"),
                    "wallpaper": ("image", "name")},
                   "board_menu":
                   {"board": ("board", "name"),
                    "menu": ("menu", "name")}}

    def __init__(self, dimension, menus, width, height, wallpaper, db=None):
        self.dimension = dimension
        self.menus = menus
        self.width = width
        self.height = height
        self.wallpaper = wallpaper
        if db is not None:
            db.boarddict[self.dimension.name] = self

    def unravel(self, db):
        self.wallpaper = db.imgdict[self.wallpaper]
        self.dimension.unravel(db)
        for menu in self.menus.itervalues():
            menu.unravel()

 
    def __eq__(self, other):
        return (
            isinstance(other, Board) and
            self.dimension == other.dimension)

    def __hash__(self):
        return self.hsh

    def getwidth(self):
        return self.width

    def getheight(self):
        return self.height

    def __repr__(self):
        return "A board, %d pixels wide by %d tall, representing the "\
            "dimension %s, containing %d spots, %d pawns, and %d menus."\
            % (self.width, self.height, self.dimension, len(self.spots),
               len(self.pawns), len(self.menus))


pull_board_cols = (
    Board.colnames["board"] +
    ["menu"] +
    Menu.valnames["menu"] +
    ["idx"] +
    MenuItem.valnames["menu_item"] +
    Style.valnames["style"] +
    Color.valnames["color"])

pull_board_qualified_cols = (
    ["board." + col for col in Board.colnames["board"]] +
    ["board_menu.menu"] +
    ["menu." + col for col in Menu.valnames["menu"]] +
    ["menu_item.idx"] +
    ["menu_item." + col for col in MenuItem.valnames["menu_item"]] +
    ["style." + col for col in Style.valnames["style"]])

pull_board_qualified_cols.sort()

pull_board_qrystr_to_style = (
    "SELECT {0} FROM board, board_menu, menu, menu_item, style"
    "WHERE board.dimension=board_menu.board "
    "AND board_menu.menu=menu.name "
    "AND menu_item.menu=menu.name "
    "AND menu.style=style.name "
    "AND ("
    "style.bg_inactive=color.name OR "
    "style.bg_active=color.name OR "
    "style.fg_inactive=color.name OR "
    "style.fg_active=color.name) "
    "AND board.dimension=?".format(", ".join(pull_board_qualified_cols)))


def load_named(db, name):
    dim = dimension.pull_named(db, name)
    db.c.execute(pull_board_qrystr, (name,))
    rows = db.c.fetchall()
    samplerow = rows.pop()
    sample = dictify_row(samplerow, pull_board_qualified_cols)
    boarddict = {
        "dimension": dim,
        "db": db,
        "width": sample["board.width"],
        "height": sample["board.height"],
        "wallpaper": sample["board.wallpaper"]}
    rows.insert(0, samplerow)
    menudict = {}
    colornames = set()
    for row in rows:
        rowdict = dictify_row(row, pull_board_qualified_cols)
        if "width" not in boarddict:
            boarddict["width"] = rowdict["board.width"]
            boarddict["height"] = rowdict["board.height"]
            boarddict["wallpaper"] = rowdict["board.wallpaper"]
        for colorfield in ["style.bg_inactive", "style.bg_active",
                           "style.fg_inactive", "style.fg_active"]:
            if rowdict[colorfield] not in db.colordict:
                Color(
                    rowdict[colorfield], rowdict["color.red"],
                    rowdict["color.green"],
                    rowdict["color.blue"], rowdict["color.alpha"],
                    db)
        if rowdict["menu.style"] not in db.styledict:
            Style(
                rowdict["menu.style"], rowdict["style.fontface"],
                rowdict["style.fontsize"],
                rowdict["style.bg_inactive"], rowdict["style.bg_active"],
                rowdict["style.fg_inactive"], rowdict["style.fg_active"],
                db)
            for colorname in [
                rowdict["style.bg_inactive"], rowdict["style.bg_active"],
                rowdict["style.fg_inactive"], rowdict["style.fg_active"]]:
                colornames.add(colorname)
        if rowdict["board_menu.menu"] not in menudict:
            menudict[rowdict["board_menu.menu"]] = Menu(
                rowdict["board_menu.menu"], rowdict["menu.left"],
                rowdict["menu.bottom"], rowdict["menu.top"],
                rowdict["menu.right"], rowdict["menu.style"],
                rowdict["menu.main_for_window"],
                rowdict["menu.visible"], db)
    color.load_named(db, iter(colornames))
    dimension.load_named(db, [boarddict["dimension"]])
    boarddict["menus"] = menudict
    return boarddict


def load_named(db, name):
    return Board(**pull_named(db, name))
