import os
import LiSE
from LiSE.allegedb import query
query.QueryEngine.path = os.path.dirname(LiSE.__file__)