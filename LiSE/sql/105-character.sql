INSERT INTO character_things (character, dimension, thing) SELECT dimension||'.Thing('||thing||')', dimension, thing FROM thing_location;
INSERT INTO character_places (character, dimension, place) SELECT dimension||'.Place('||name||')', dimension, name FROM place;
INSERT INTO character_portals (character, dimension, origin, destination) SELECT dimension||'.Portal('||origin||'->'||destination||')', dimension, origin, destination FROM portal;
INSERT INTO character_things (character, dimension, thing) VALUES ('household', 'Physical', 'me'), ('household', 'Physical', 'mom');
INSERT INTO character_places (character, dimension, place) SELECT 'household', 'Physical', name FROM place WHERE dimension='Physical';
INSERT INTO character_portals (character, dimension, origin, destination) SELECT 'household', 'Physical', origin, destination FROM portal WHERE dimension='Physical';
INSERT INTO character_stats (character, stat, value) VALUES
       ('Physical.Thing(me)', 'weight', '170'),
       ('Physical.Thing(mom)', 'weight', '200'),
       ('Physical.Thing(me)', 'dexterity', '9'),
       ('Physical.Thing(mom)', 'dexterity', '10');
INSERT INTO character_stats (character, stat, value) VALUES
       ('household', 'income', '10000'),
       ('household', 'mood', 'stable');
