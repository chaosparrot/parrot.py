from lib.default_config import *
import math
import os
_user_config_file = CODE_FOLDER + "/config.py"
if not os.path.exists(_user_config_file):
    configfile = open(_user_config_file, "w")
    configfile.write('DEFAULT_CLF_FILE = ""\n')
    configfile.write('STARTING_MODE = ""\n')
    configfile.write('MICROPHONE_SEPARATOR = None\n')
    configfile.close()
with open(_user_config_file, encoding="utf-8") as _f:
    exec(compile(_f.read(), _user_config_file, "exec"))

# Define constants here that should be set after the user config
MS_PER_FRAME = math.floor(RECORD_SECONDS / SLIDING_WINDOW_AMOUNT * 1000)
