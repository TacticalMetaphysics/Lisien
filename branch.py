from util import SaveableMetaclass


__metaclass__ = SaveableMetaclass


class Branch:
    tables = [(
        "branch",
        {"idx": "integer not null",
         "parent": "integer not null",
         "start": "integer not null",
         "parm": "text"},
        ("idx",),
        {"parent": ("branch", "idx")},
        ["parent<idx"])]
    def __init__(self, rumor, i, start, parm):
        self.rumor = rumor
        self.i = i
        self.start = start
        self.parm = parm
        self.children = []

    def __int__(self):
        return self.i

    def get_tabdict(self):
        return {
            "branch": {
                "idx": self.i,
                "start": self.start,
                "parm": self.parm}}

    def save(self):
        for child in self.children:
            child.save()
        self.coresave()
