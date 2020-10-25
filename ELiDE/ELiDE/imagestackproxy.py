from kivy.clock import Clock
from kivy.properties import ObjectProperty

from ELiDE.kivygarden.texturestack import ImageStack
from .util import trigger


class ImageStackProxy(ImageStack):
    proxy = ObjectProperty()
    name = ObjectProperty()

    def finalize(self, initial=True):
        if getattr(self, '_finalized', False):
            return
        if (
                self.proxy is None or
                not hasattr(self.proxy, 'name')
        ):
            Clock.schedule_once(self.finalize, 0)
            return
        if initial:
            self.name = self.proxy.name
            if '_image_paths' in self.proxy:
                try:
                    self.paths = self.proxy['_image_paths']
                except Exception as ex:
                    if not ex.args[0].startswith('Unable to load image type'):
                        raise ex
                    self.paths = self.default_image_paths
            else:
                self.paths = self.proxy.setdefault(
                    '_image_paths', self.default_image_paths
                )
        self.bind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys
        )
        self._finalized = True
        self.finalize_children()

    def finalize_children(self, *args):
        for child in self.children:
            child.finalize()
        self.bind(children=self._trigger_finalize_children)

    _trigger_finalize_children = trigger(finalize_children)

    def unfinalize(self):
        self.unbind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys
        )
        self._finalized = False

    def pull_from_proxy(self, *args):
        initial = not hasattr(self, '_finalized')
        self.unfinalize()
        for key, att in [
                ('_image_paths', 'paths'),
                ('_offxs', 'offxs'),
                ('_offys', 'offys')
        ]:
            if key in self.proxy and self.proxy[key] != getattr(self, att):
                setattr(self, att, self.proxy[key])
        self.finalize(initial)

    def _trigger_pull_from_proxy(self, *args, **kwargs):
        if hasattr(self, '_scheduled_pull_from_proxy'):
            Clock.unschedule(self._scheduled_pull_from_proxy)
        self._scheduled_pull_from_proxy = Clock.schedule_once(self.pull_from_proxy, 0)

    @trigger
    def _trigger_push_image_paths(self, *args):
        self.proxy['_image_paths'] = list(self.paths)

    @trigger
    def _trigger_push_offxs(self, *args):
        self.proxy['_offxs'] = list(self.offxs)

    @trigger
    def _trigger_push_offys(self, *args):
        self.proxy['_offys'] = list(self.offys)

    @trigger
    def _trigger_push_stackhs(self, *args):
        self.proxy['_stackhs'] = list(self.stackhs)

    @trigger
    def restack(self, *args):
        childs = sorted(list(self.children), key=lambda child: child.priority, reverse=True)
        self.clear_widgets()
        for child in childs:
            self.add_widget(child)
        self.do_layout()