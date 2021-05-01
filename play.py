import lib.config_generator
from config.config import *
from lib.listen import start_nonblocking_listen_loop, load_running_classifier
from lib.mode_switcher import ModeSwitcher
import sys, getopt
import lib.ipc_manager as ipc_manager

def main(argv):
    if (ipc_manager.getParrotState() != "not_running"):
        print( "Parrot might already be running somewhere, stopping that instance..." )
        ipc_manager.requestParrotState("stopped")

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
    
    classifier = load_running_classifier(default_classifier_file)
    mode_switcher = ModeSwitcher( input_testing_mode )
    ipc_manager.requestParrotState("running")
    ipc_manager.setParrotState("running")    
    mode_switcher.switchMode( starting_mode )
    
    start_nonblocking_listen_loop( classifier, mode_switcher, SAVE_REPLAY_DURING_PLAY, SAVE_FILES_DURING_PLAY, -1, True )
    mode_switcher.exit()
    ipc_manager.close()

if __name__ == "__main__":
   main(sys.argv[1:])