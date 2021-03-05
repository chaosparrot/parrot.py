import numpy as np
from config.config import *
import pyaudio
import time
import msvcrt
from queue import *
import lib.ipc_manager as ipc_manager
DISCONNECTION_DETECTION_THRESHOLD = 1.0

LOOP_STATE_CONTINUE = 1 # Continue the listening loop
LOOP_STATE_BREAK = -1 # Break out and stop the current loop
LOOP_STATE_BLOCK = 0 # Block the listening loop and just wait for state transitions
LOOP_STATE_SWITCHED = 2 # Special loop state where the classifier or the mode has been switched - This requires additional setup to make work
poll_counter = 0

def keypress_state_change():
    ESCAPEKEY = b'\x1b'
    SPACEBAR = b' '

    requested_state = False
    if( msvcrt.kbhit() ):
        character = msvcrt.getch()
        if (character == SPACEBAR):
            requested_state = "paused" if ipc_manager.getParrotState() not in ["paused", "switching"] else "running"
        elif ( character == ESCAPEKEY ):
            requested_state = "stopped"
    
    return requested_state

def set_loop_state( state ):
    ipc_manager.requestParrotState(state)
    ipc_manager.setParrotState(state)

# Detect state transitions by checking the IPC requested state
# Then detecting the keypress state if no requested state is found
# And in the final check - Detect a disconnection
def detect_state_transition(current_state, listening_state, currenttime):
    requested_state = ipc_manager.getRequestedParrotState()
    if (requested_state == False):
        requested_state = keypress_state_change()
    if (current_state not in ["disconnected", "paused"] and requested_state == False and currenttime - listening_state['last_audio_update'] > DISCONNECTION_DETECTION_THRESHOLD):
        requested_state = "disconnected"
    return requested_state 

# Detects a transitioning state and acts accordingly
def transition_state(listening_state, modeSwitcher, current_state, requested_state = False):

    # Any state can transition to switch and then transition back
    if (requested_state == "switching" or requested_state == "switch_and_run"):
        if (listening_state['stream']):
            listening_state['stream'].stop_stream()
        if( listening_state['audioQueue'] != None ):
            listening_state['audioQueue'].queue.clear()
        
        modeSwitcher.switchMode(ipc_manager.getMode(), requested_state == "switch_and_run")
        if ( listening_state['classifier_name'] != ipc_manager.getClassifier() ):
            print( "Switching classifier to " + ipc_manager.getClassifier() )
            print( "Listening stopped" )
            print( "-----------------" )
            listening_state['classifier_name'] = ipc_manager.getClassifier()            
            listening_state['stream'].stop_stream()
            set_loop_state(current_state)
            
            # Reset the stream
            listening_state['restart_listen_loop'] = True
        return LOOP_STATE_SWITCHED
    
    # Running can transition to paused and disconnected
    if (current_state == "running" ):
        if (requested_state == False):    
            return LOOP_STATE_CONTINUE
        elif( requested_state == "paused"):
            print( "Listening paused!" )
            if (listening_state['stream']):
                listening_state['stream'].stop_stream()
        elif ( requested_state == "disconnected" ):
            if (listening_state['stream']):
                listening_state['stream'].stop_stream()
            if( listening_state['audioQueue'] != None ):
                listening_state['audioQueue'].queue.clear()
            print( "Found a stall in audio updates" )
            print( "Assuming a mic disconnection, beginning to poll mic connection" )
            
    # A disconnected state can be in recovering mode, or in a paused state where it does not poll for connections
    if (current_state == "disconnected"):
        global poll_counter
        if (requested_state == False):
            audio = pyaudio.PyAudio()        
            try:
                stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                        input_device_index=INPUT_DEVICE_INDEX,
                        frames_per_buffer=round( RATE * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT ))
                print( "" )
                print( "Did not receive errors during reconnection to mic, restarting stream" )
                if ( stream is not None ):
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()
                    
                # Reset the stream
                listening_state['restart_listen_loop'] = True
                set_loop_state("running")
                return LOOP_STATE_BREAK
            except Exception as e:
                poll_counter = poll_counter + 1
                audio.terminate()
                print( "Reconnection attempt " + str( poll_counter ) + "...", end="\r")
        elif (requested_state == "paused"):
            print( "" )
            print( "Reconnecting paused" )
            set_loop_state("paused")
            poll_counter = 0
        elif (requested_state == "stopped"):
            print( "" )
            
    # A paused state can resume listening or transition to a disconnected state
    if (current_state == "paused"):
        if (requested_state == "running" ):
            print( "Listening resumed!" )
            if (listening_state['stream']):
                try:
                    listening_state['stream'].start_stream()
                    set_loop_state(requested_state)                    
                    return LOOP_STATE_CONTINUE                    
                except IOError as e:
                    print( "An error occured during the resuming of the listening")
                    return LOOP_STATE_CONTINUE
    
    # Any state can be stopped to terminate Parrot
    if (requested_state == "stopped"):
        listening_state['currently_recording'] = False
        print( "Listening stopped" )
        return LOOP_STATE_BREAK

    return LOOP_STATE_BLOCK


# Manages the state machine for listening
# By detecting changes in connectivity status, key presses and ipc requests
def manage_loop_state(current_state, listening_state, modeSwitcher=None, currenttime=0, STATE_POLLING_THRESHOLD = 0.1):
    requested_state = detect_state_transition(current_state, listening_state, currenttime)
    loop_state = transition_state(listening_state, modeSwitcher, current_state, requested_state)
    
    # After switching modes or classifiers, do not set the loop state, this is handled by the mode switcher instead
    if (loop_state == LOOP_STATE_SWITCHED ):
        loop_state = transition_state(listening_state, modeSwitcher, "paused", ipc_manager.getParrotState())
    
    if (loop_state == LOOP_STATE_BLOCK ):
        set_loop_state(requested_state)
        time.sleep( STATE_POLLING_THRESHOLD )
        return manage_loop_state(ipc_manager.getParrotState(), listening_state, modeSwitcher, currenttime, STATE_POLLING_THRESHOLD)
        
    return loop_state == LOOP_STATE_CONTINUE and listening_state['restart_listen_loop'] == False