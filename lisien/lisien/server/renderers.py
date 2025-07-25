from rest_framework.renderers import BaseRenderer


class LiSEMessagePackRenderer(BaseRenderer):
    meda_type = "application/msgpack"
    format = "msgpack"
    render_style = "binary"
    charset = None

    def render(self, data, accepted_media_type=None, renderer_context=None) -> bytes:
        if data is None:
            return b''
        if renderer_context is None or "LiSE" not in renderer_context:
            raise RuntimeError("No LiSE engine")
        eng = renderer_context["LiSE"]
        return eng.pack(data)