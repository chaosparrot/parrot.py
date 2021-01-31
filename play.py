from config.config import *
import joblib
from lib.listen import start_nonblocking_listen_loop
from lib.mode_switcher import ModeSwitcher
from lib.audio_model import AudioModel
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
    if( default_classifier_file != "dummy" ):
        print( "Loading classifier " + CLASSIFIER_FOLDER + "/" + default_classifier_file + ".pkl" )
        classifier = joblib.load( CLASSIFIER_FOLDER + "/" + default_classifier_file + ".pkl" )
        
        if( not isinstance( classifier, AudioModel ) ):
            settings = {
                'version': 0,
                'RATE': RATE,
                'CHANNELS': CHANNELS,
                'RECORD_SECONDS': RECORD_SECONDS,
                'SLIDING_WINDOW_AMOUNT': SLIDING_WINDOW_AMOUNT,
                'feature_engineering': None
            }
            
            classifier = AudioModel( settings, classifier )
    else:
        print( "Loading dummy classifier for testing purposes" )
        from lib.dummy_classifier import DummyClassifier
        classifier = DummyClassifier()

    mode_switcher = ModeSwitcher( input_testing_mode )
    mode_switcher.switchMode( starting_mode )
    start_nonblocking_listen_loop( classifier, mode_switcher, SAVE_REPLAY_DURING_PLAY, SAVE_FILES_DURING_PLAY, -1, True )
    mode_switcher.exit()

if __name__ == "__main__":
   main(sys.argv[1:])