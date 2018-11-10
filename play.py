from config.config import *
from sklearn.externals import joblib
from lib.listen import start_listen_loop
from lib.mode_switcher import ModeSwitcher

# Load the trained classifier
classifier = joblib.load( CLASSIFIER_FOLDER + "/" + DEFAULT_CLF_FILE + ".pkl" )
print( "Loaded classifier " + CLASSIFIER_FOLDER + "/" + DEFAULT_CLF_FILE + ".pkl" )
	
mode_switcher = ModeSwitcher()
mode_switcher.switchMode( STARTING_MODE )
start_listen_loop( classifier, mode_switcher, True )