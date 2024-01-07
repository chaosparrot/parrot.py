from config.config import *
import pyaudio
import time
import math
import os
import glob
from queue import *
import threading
import traceback
import sys
from lib.listen import validate_microphone_input
from lib.key_poller import KeyPoller
from lib.print_status import get_current_status, reset_previous_lines, clear_previous_lines
from lib.typing import DetectionLabel, DetectionState
from lib.stream_processing import CURRENT_VERSION, CURRENT_DETECTION_STRATEGY
from lib.typing import DetectionState, DetectionFrame
from lib.stream_recorder import StreamRecorder
from lib.srt import count_total_label_ms, ms_to_srt_timestring
from typing import List

# Countdown from seconds to 0
def countdown( seconds ):
    with KeyPoller() as key_poller:
        for i in range( -seconds, 0 ):
            print("recording in... " + str(abs(i)), end="\r")
            time.sleep( 1 )
            if( record_controls(key_poller) == False ):
                return False;
    print("                          ", end="\r")
    return True

def record_controls( key_poller, recordQueue=None ):
    global currently_recording
    global recorders
    ESCAPEKEY = '\x1b'
    SPACEBAR = ' '
    BACKSPACE = '\x08'
    MINUS = '-'
    
    character = key_poller.poll()
    if(character is not None):
        # Clear the last 3 seconds if backspace was pressed
        if character == BACKSPACE or character == MINUS:
            if (recorders is not None):
                main_state = None
                secondary_states = []
                for mic_index in recorders:
                    if main_state is None:
                        main_state = recorders[mic_index].get_detection_state()
                    else:
                        secondary_states.append(recorders[mic_index].get_detection_state())                
                    recorders[mic_index].pause()
                should_resume = False
                
                # Clear and update the detection states
                index = 0
                if main_state is not None:
                    main_state.state = "deleting"
                    print_status(main_state, secondary_states)
                
                for mic_index in recorders:
                    should_resume = recorders[mic_index].clear(3)
                    if index == 0:
                        main_state = recorders[mic_index].get_detection_state()
                    else:
                        secondary_states[index - 1] = recorders[mic_index].get_detection_state()
                    index += 1
                    print_status(main_state, secondary_states)

                if main_state is not None:
                    main_state.state = "recording"
                    print_status(main_state, secondary_states)
                
                # Wait for the sound of the space bar to dissipate before continuing recording
                time.sleep(0.3)
                if should_resume:
                    for mic_index in recorders:
                        recorders[mic_index].resume()
        elif( character == ESCAPEKEY ):
            currently_recording = False
            return False

        elif character == SPACEBAR:
            if( recordQueue == None ):
                print( "Recording paused!" )

            main_state = None
            secondary_states = []
            if (recorders is not None):
                for mic_index in recorders:
                    if main_state is None:
                        main_state = recorders[mic_index].get_detection_state()
                    else:
                        secondary_states.append(recorders[mic_index].get_detection_state())
                    recorders[mic_index].pause()
                    recorders[mic_index].reset_label_count()
                    
                # Do post processing and printing of the status
                if main_state is not None:
                    index = 0
                    main_state.state = "deleting"
                    print_status(main_state, secondary_states)
                    
                    for mic_index in recorders:
                        recorders[mic_index].post_processing(
                            lambda internal_progress, state, extra=secondary_states: print_status(main_state, extra)
                        )
                        
                        # Update the states so the numbers count up nicely
                        if index == 0:
                            main_state = recorders[mic_index].get_detection_state()
                        else:
                            secondary_states[index - 1] = recorders[mic_index].get_detection_state()
                        index += 1

                    main_state.state = "paused"
                    print_status(main_state, secondary_states)

            # Pause the recording by looping until we get a new keypress
            while( True ):
                # If the audio queue exists - make sure to clear it continuously
                if( recordQueue != None ):
                    for key in recordQueue:
                        recordQueue[key].queue.clear()

                character = key_poller.poll()
                if character is not None:
                    if character == SPACEBAR:
                        if main_state is not None:
                            main_state.state = "recording"
                            print_status(main_state, secondary_states)
                        
                        # Wait for the sound of the space bar to dissipate before continuing recording
                        time.sleep(0.3)
                        if recorders is not None:
                            for mic_index in recorders:
                                recorders[mic_index].resume()
                        return True
                    # Clear the last 3 seconds if backspace was pressed
                    elif character == BACKSPACE or character == MINUS:
                        if recorders is not None and main_state is not None:
                            index = 0
                            for mic_index in recorders:
                                recorders[mic_index].clear(3)
                                if index == 0:
                                    main_state = recorders[mic_index].get_detection_state()
                                else:
                                    secondary_states[index - 1] = recorders[mic_index].get_detection_state()
                                index += 1
                                print_status(main_state, secondary_states)
                            main_state.state = "paused"
                            print_status(main_state, secondary_states)
                    
                    # Stop the recording session
                    elif character == ESCAPEKEY:
                        currently_recording = False
                        return False
                time.sleep(0.3)
    return True    
    
def record_sound():
    audio = pyaudio.PyAudio()

    print( "-------------------------" )
    print( "Let's record some sounds!")
    print( "This script is going to listen to your microphone input" )
    print( "And record tiny audio files to be used for learning later" )
    print( "-------------------------" )
    
    ms_per_frame = math.floor(RECORD_SECONDS / SLIDING_WINDOW_AMOUNT * 1000)
    directory_counts = {}
    try:
        if os.path.exists(RECORDINGS_FOLDER):
            glob_path = RECORDINGS_FOLDER + "/*/"
            existing_dirs = glob.glob(glob_path)
            if existing_dirs:
                print("")
                print("These sounds already have a folder:")
            for dirname in existing_dirs:
                # cut off glob path, but leave two more characters
                # at the start to account for */
                # also remove the trailing slash
                directory_name = dirname[len(glob_path) - 2:-1]

                # Count the currently recorded amount of data
                current_count = count_total_label_ms(directory_name, os.path.join(RECORDINGS_FOLDER, directory_name), ms_per_frame)
                directory_counts[directory_name] = current_count
                time_recorded = " ( " + ms_to_srt_timestring(current_count, False).split(",")[0] + " )"
                
                print(" - ", directory_name.ljust(30) + time_recorded )
            print("")
            print("NOTE: It is recommended to record roughly the same amount for each sound")
            print("As it will improve the ability for the machine learning models to learn from the data")
            print("")            
    except:
        # Since this is just a convenience feature, exceptions shall not
        # cause recording to abort, whatever happens
        pass

    directory = input("Whats the name of the sound are you recording? ")
    while (directory == ""):
        directory = input("")
    
    if not os.path.exists(RECORDINGS_FOLDER + "/" + directory):
        os.makedirs(RECORDINGS_FOLDER + "/"  + directory)
    if not os.path.exists(RECORDINGS_FOLDER + "/" + directory + "/source"):
        os.makedirs(RECORDINGS_FOLDER + "/"  + directory + "/source")
    if not os.path.exists(RECORDINGS_FOLDER + "/" + directory + "/segments"):
        os.makedirs(RECORDINGS_FOLDER + "/"  + directory + "/segments")

    # Note - this assumes a maximum of 10 possible input devices, which is probably wrong but eh
    print("What microphone do you want to record with? ( Empty is the default system mic, [X] exits the recording menu )")
    print("You can put a space in between numbers to record with multiple microphones")
    for index in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(index)
        if (device_info and device_info['name'] and device_info['maxInputChannels'] > 0):
            default_mic = " - " if index != INPUT_DEVICE_INDEX else " DEFAULT - "
            host_api = audio.get_host_api_info_by_index(device_info['hostApi'])
            host_api_string = " " + host_api["name"] if host_api else ""
            print("[" + str(index) + "]" + default_mic + device_info['name'] + host_api_string)
                
    mic_index_string = input("")
    mic_indecis = []
    if mic_index_string == "":
        mic_indecis = [str(INPUT_DEVICE_INDEX)]
    elif mic_index_string.strip().lower() == "x":
        return;
    else:
        mic_indecis = mic_index_string.split()
    valid_mics = []
    for mic_index in mic_indecis:
        if (str.isdigit(mic_index) and validate_microphone_index(audio, int(mic_index))):
            valid_mics.append(int(mic_index))
    
    if len(valid_mics) == 0:
        print("No usable microphones selected - Exiting")
        return;

    print("")
    print("Record keyboard controls:")
    print("[SPACE] is used to pause and resume the recording session")
    print("[BACKSPACE] or [-] removes the last 3 seconds of the recording")
    print("[ESC] stops the current recording")
    print("")    

    global recordQueue
    global recorders
    recorders = {}
    recordQueue = {}
    labels = {}
    labels[directory] = directory_counts[directory] if directory in directory_counts else 0    
    
    if( countdown( 5 ) == False ):
        return;
    
    global currently_recording
    currently_recording = True
    
    time_string = str(int(time.time()))
    for index, microphone_index in enumerate(valid_mics):
        FULL_WAVE_OUTPUT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/source/mici_" + str(microphone_index) + "__" + time_string + ".wav"
        SRT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/segments/mici_" + str(microphone_index) + "__" + time_string + ".v" + str(CURRENT_VERSION) + ".srt"
        non_blocking_record(labels, FULL_WAVE_OUTPUT_FILENAME, SRT_FILENAME, microphone_index, index==0)
    
    # wait for stream to finish
    while currently_recording == True:
        time.sleep(0.1)
    
    main_state = None
    secondary_states = []
    for mic_index in recorders:
        if main_state is None:
            main_state = recorders[mic_index].get_detection_state()
        else:
            secondary_states.append(recorders[mic_index].get_detection_state())
        recorders[mic_index].pause()

    index = 0
    for mic_index in recorders:
        callback = None if currently_recording == -1 else lambda internal_progress, state, extra=secondary_states: print_status(main_state, extra)
        recorders[mic_index].stop(
            callback
        )
        
        # Update the states so the numbers count up nicely
        if index == 0:
            main_state = recorders[mic_index].get_detection_state()
        else:
            secondary_states[index - 1] = recorders[mic_index].get_detection_state()
        index += 1
    
    if currently_recording != -1:    
        main_state.state = "processed"
        print_status(main_state, secondary_states)    

# Consumes the recordings in a sliding window fashion - Always combining the two latest chunks together    
def record_consumer(labels, FULL_WAVE_OUTPUT_FILENAME, SRT_FILE, MICROPHONE_INPUT_INDEX, print_stuff=False):
    global recordQueue
    global currently_recording
    global recorders
    mic_index = 'index' + str(MICROPHONE_INPUT_INDEX)
    indexedQueue = recordQueue[mic_index]
    recorder = recorders[mic_index]

    if print_stuff:
        current_status = recorder.get_status()
        for line in current_status:
            print( line )
    
    try:
        with KeyPoller() as key_poller:
            while( currently_recording ):
                while( not indexedQueue.empty() ):
                    recorder.add_audio_frame(indexedQueue.get())
                    
                    if print_stuff:
                        extra_states = []
                        for recorder_mic_index in recorders:
                            if mic_index != recorder_mic_index and recorder:
                                extra_states.append(recorders[recorder_mic_index].get_detection_state())
                        current_status = recorder.get_status(extra_states)
                        reset_previous_lines(len(current_status))
                        for line in current_status:
                            print( line )
                
                # Only listen for keys in the main listener
                if print_stuff:
                    record_controls( key_poller, recordQueue )

                time.sleep(0.001)
                
    except Exception as e:
        print( "----------- ERROR DURING RECORDING -------------- " )
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        currently_recording = -1

def multithreaded_record( in_data, frame_count, time_info, status, queue ):
    queue.put( in_data )
    
    return in_data, pyaudio.paContinue
                
# Records a non blocking audio stream and saves the source and SRT file for it
def non_blocking_record(labels, FULL_WAVE_OUTPUT_FILENAME, SRT_FILE, MICROPHONE_INPUT_INDEX, print_logs):
    global recordQueue
    global recorders
    
    mic_index = 'index' + str(MICROPHONE_INPUT_INDEX)

    recordQueue[mic_index] = Queue(maxsize=0)
    micindexed_lambda = lambda in_data, frame_count, time_info, status, queue=recordQueue[mic_index]: multithreaded_record(in_data, frame_count, time_info, status, queue)
    
    detection_strategy = CURRENT_DETECTION_STRATEGY
    ms_per_frame = math.floor(RECORD_SECONDS / SLIDING_WINDOW_AMOUNT * 1000)
    detection_labels = []
    for label in list(labels.keys()):
        detection_labels.append(DetectionLabel(label, 0, labels[label], "", 0, 0, 0, 0))
    
    audio = pyaudio.PyAudio()

    recorders[mic_index] = StreamRecorder(
        audio,
        audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            input_device_index=MICROPHONE_INPUT_INDEX,
            frames_per_buffer=round( RATE * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT ),
            stream_callback=micindexed_lambda),
        FULL_WAVE_OUTPUT_FILENAME,
        SRT_FILE,
        DetectionState(detection_strategy, "recording", ms_per_frame, 0, True, 0, 0, 0, detection_labels)
    )

    consumer = threading.Thread(name='consumer', target=record_consumer, args=(labels, FULL_WAVE_OUTPUT_FILENAME, SRT_FILE, MICROPHONE_INPUT_INDEX, print_logs))
    consumer.setDaemon( True )
    consumer.start()
    recorders[mic_index].resume()

def print_status(detection_state: DetectionState, extra_states: List[DetectionState]):
    current_status = get_current_status(detection_state, extra_states)
    reset_previous_lines(len(current_status))
    for line in current_status:
        print( line )

def validate_microphone_index(audio, input_index):
    micDict = {'name': 'Missing Microphone index ' + str(input_index)}
    try:
        micDict = audio.get_device_info_by_index( input_index )
        if (micDict and micDict['maxInputChannels'] > 0):            
            host_api = audio.get_host_api_info_by_index(micDict['hostApi'])
            host_api_string = " " + host_api["name"] if host_api else ""
            print( "Using input from " + micDict['name'] + host_api_string )
            return True
        else:
            raise IOError( "Invalid number of channels" )
    except IOError as e:
        print("Could not connect enough audio channels to " + micDict['name'] + ", disabling this mic for recording")
        return False
