"""
This script sets up the environment and runs the ELiDE application.
It adds the LiSE and ELiDE directories to the system path, defines a function to get the application configuration,
and attempts to import and run the ELiDE application.
"""

import os
import sys
from multiprocessing import freeze_support

wd = os.getcwd()
sys.path.extend([wd + "/LiSE", wd + "/ELiDE"])

# _args follows the Python convention to indicate unused variables.
def get_application_config(_args):  
    return wd + "/ELiDE.ini"

if __name__ == "__main__":
    freeze_support()

    # Try to import ELiDE
    # if not found, print a message
    try:
        from ELiDE.app import ELiDEApp
    except ImportError:
        print("ELiDE module not found. Please install it.")

    app = ELiDEApp()
    app.get_application_config = get_application_config
    app.run()
