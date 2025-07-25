from rest_framework.exceptions import ParseError
from rest_framework.parsers import BaseParser


class LiSEMessagePackParser(BaseParser):
    media_type = "application/msgpack"

    def parse(self, stream, media_type=None, parser_context=None):
        if parser_context is None or "LiSE" not in parser_context:
            raise RuntimeError("No LiSE engine")
        eng = parser_context["LiSE"]
        try:
            return eng.unpack(stream)
        except Exception as ex:
            raise ParseError(ex.args[0]) from ex