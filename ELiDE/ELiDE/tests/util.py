from blinker import Signal
from kivy.input.motionevent import MotionEvent


class MockTouch(MotionEvent):
    def depack(self, args):
        self.is_touch = True
        self.sx = args['sx']
        self.sy = args['sy']
        super().depack(args)


class ListenableDict(dict, Signal):
    def __init__(self):
        Signal.__init__(self)


class MockEngine(Signal):
    universal = ListenableDict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn = 0
        self._ready = True

    def __setattr__(self, key, value):
        if not hasattr(self, '_ready'):
            super().__setattr__(key, value)
            return
        self.send(self, key=key, value=value)
        super().__setattr__(key, value)

    def next_turn(self, *args, **kwargs):
        self.turn += 1
        kwargs['cb']('next_turn', 'master', self.turn, 0, ([], {}))