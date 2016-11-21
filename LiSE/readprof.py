import pstats
import sys

stats = pstats.Stats()
if len(sys.argv) > 1:
    stats.load_stats(sys.argv[-1])
else:
    stats.load_stats('dump.prof')
stats.sort_stats('cumtime').reverse_order().print_stats()
