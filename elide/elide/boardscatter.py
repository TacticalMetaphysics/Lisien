from kivy.graphics.transformation import Matrix
from kivy.uix.scatter import ScatterPlane


class BoardScatterPlane(ScatterPlane):
	def on_touch_down(self, touch):
		if touch.is_mouse_scrolling:
			scale = self.scale + (
				0.05 if touch.button == "scrolldown" else -0.05
			)
			if (self.scale_min and scale < self.scale_min) or (
				self.scale_max and scale > self.scale_max
			):
				return
			rescale = scale * 1.0 / self.scale
			self.apply_transform(
				Matrix().scale(rescale, rescale, rescale),
				post_multiply=True,
				anchor=self.to_local(*touch.pos),
			)
			return self.dispatch("on_transform_with_touch", touch)
		return super().on_touch_down(touch)

	def apply_transform(self, trans, post_multiply=False, anchor=(0, 0)):
		super().apply_transform(
			trans, post_multiply=post_multiply, anchor=anchor
		)
		self._last_transform = trans, post_multiply, anchor

	def on_transform_with_touch(self, touch):
		x, y = self.pos
		w = self.board.width * self.scale
		h = self.board.height * self.scale
		if hasattr(self, "_last_transform") and (
			w < self.parent.width or h < self.parent.height
		):
			trans, post_multiply, anchor = self._last_transform
			super().apply_transform(trans.inverse(), post_multiply, anchor)
			return
		if x > self.parent.x:
			self.x = self.parent.x
		if y > self.parent.y:
			self.y = self.parent.y
		if x + w < self.parent.right:
			self.x = self.parent.right - w
		if y + h < self.parent.top:
			self.y = self.parent.top - h
