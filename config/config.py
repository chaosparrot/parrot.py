from lib.default_config import *
import os
if not os.path.exists('data/code/config.py'):
    configfile = open("data/code/config.py", "w")
    configfile.write('DEFAULT_CLF_FILE = ""\n')
    configfile.write('STARTING_MODE = ""\n')
    configfile.close()
from data.code.config import *