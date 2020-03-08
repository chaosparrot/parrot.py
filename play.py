from config.config import *
import joblib
from lib.listen import start_listen_loop, start_nonblocking_listen_loop
from lib.mode_switcher import ModeSwitcher
import sys, getopt

def main(argv):
    # Process the optional flags
    opts, args = getopt.getopt(argv,"tc:m:",["testing:classifier=:mode="])
       
    input_testing_mode = INPUT_TESTING_MODE
    default_classifier_file = DEFAULT_CLF_FILE
    starting_mode = STARTING_MODE
    for opt, arg in opts:
        if opt in ("-c", "--classifier"):
            default_classifier_file = arg
        elif opt in ("-t", "--testing"):
            input_testing_mode = True
            print( "Enabling testing mode - No inputs will be sent to the keyboard and mouse!" )
        elif opt in ("-m", "--mode"):
            starting_mode = arg
    
    # Load the trained classifier
    print( "Loading classifier " + CLASSIFIER_FOLDER + "/" + default_classifier_file + ".pkl" )
    classifier = joblib.load( CLASSIFIER_FOLDER + "/" + default_classifier_file + ".pkl" )
            
    mode_switcher = ModeSwitcher( input_testing_mode )
    mode_switcher.switchMode( starting_mode )
    start_nonblocking_listen_loop( classifier, mode_switcher, SAVE_REPLAY_DURING_PLAY, SAVE_FILES_DURING_PLAY, -1, True )
    mode_switcher.exit()

if __name__ == "__main__":
   main(sys.argv[1:])