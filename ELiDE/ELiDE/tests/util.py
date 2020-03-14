from kivy.input.motionevent import MotionEvent


class TestTouch(MotionEvent):
    def depack(self, args):
        self.is_touch = True
        self.sx = args['sx']
        self.sy = args['sy']
        super().depack(args)