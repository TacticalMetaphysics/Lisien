INSERT INTO character_things (character, dimension, thing) SELECT dimension||'.Thing('||thing||')', dimension, thing FROM thing_location;
INSERT INTO character_places (character, dimension, place) SELECT dimension||'.Place('||name||')', dimension, name FROM place;
INSERT INTO character_portals (character, dimension, origin, destination) SELECT dimension||'.Portal('||origin||'->'||destination||')', dimension, origin, destination FROM portal;
