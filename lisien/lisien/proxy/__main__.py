import os
import sys

from . import EngineProcessManager

if __name__ == "__main__":
	if os.path.exists(sys.argv[-1]) and os.path.isfile(sys.argv[-1]):
		mgr = EngineProcessManager()
		eng = mgr.start(replay_file=sys.argv[-1], loglevel="debug")
		mgr.shutdown()
