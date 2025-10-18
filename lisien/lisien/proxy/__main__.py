import os
import sys

from lisien.proxy.manager import EngineProxyManager

if __name__ == "__main__":
	if os.path.exists(sys.argv[-1]) and os.path.isfile(sys.argv[-1]):
		mgr = EngineProxyManager()
		eng = mgr.start(replay_file=sys.argv[-1], loglevel="debug")
		mgr.shutdown()
