--This file is part of LiSE, a toolkit for life simulators.
--Copyright (C) Zachary Spector 2013
UPDATE img SET stacking_height=8 WHERE name IN ('blind', 'brownstone', 'orange');
UPDATE img SET stacking_height=12 WHERE name IN ('block', 'brutalist');
UPDATE img SET stacking_height=6 WHERE name IN ('crossroad', 'enterprise', 'monolith', 'olivine', 'soviet');
UPDATE img SET stacking_height=5 WHERE name IN ('sidewalk', 'street-ne-sw', 'street-nw-se');
UPDATE img SET stacking_height=4 WHERE name='spacer';
