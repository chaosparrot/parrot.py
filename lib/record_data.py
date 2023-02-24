from config.config import *
import pyaudio
import wave
import time
from time import sleep
import scipy.io.wavfile
import audioop
import math
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.signal import blackmanharris
from lib.machinelearning import get_loudest_freq, get_recording_power
import os
import glob
from queue import *
import threading
import traceback
import sys
from lib.listen import validate_microphone_input
from lib.key_poller import KeyPoller
from lib.stream_processing import CURRENT_VERSION, CURRENT_DETECTION_STRATEGY, process_audio_frame
from lib.srt import persist_srt_file
from lib.print_status import get_current_status, reset_previous_lines, clear_previous_lines
from lib.typing import DetectionLabel, DetectionState
import struct
lock = threading.Lock()

# Countdown from seconds to 0
def countdown( seconds ):
    with KeyPoller() as key_poller:
        for i in range( -seconds, 0 ):
            print("recording in... " + str(abs(i)), end="\r")
            sleep( 1 )
            if( record_controls(key_poller) == False ):
                return False;
    print("                          ", end="\r")
    return True

def record_controls( key_poller, recordQueue=None ):
    global currently_recording
    global streams
    global lock
    ESCAPEKEY = '\x1b'
    SPACEBAR = ' '
    
    character = key_poller.poll()
    if(character is not None):    
        if( character == SPACEBAR ):
            print( "Recording paused!" )
            with lock:
                if( recordQueue != None ):
                    recordQueue['status'] = 'paused'

            if (streams is not None):
                for stream in streams:
                    streams[stream].stop_stream()

            # Pause the recording by looping until we get a new keypress
            while( True ):
                
                ## If the audio queue exists - make sure to clear it continuously
                if( recordQueue != None ):
                    for key in recordQueue:
                        recordQueue[key].queue.clear()
                
                character = key_poller.poll()
                if(character is not None and ( recordQueue is None or recordQueue['status'] != 'processing') ):
                    if( character == SPACEBAR ):
                        with lock:
                            if( recordQueue != None ):
                                recordQueue['status'] = 'recording'

                        if (streams is not None):
                            for stream in streams:
                                streams[stream].start_stream()
                        return True
                    elif( character == ESCAPEKEY ):
                        currently_recording = False
                        return False
                time.sleep(0.3)
        elif( character == ESCAPEKEY ):
            currently_recording = False
            return False            
    return True    
    
def record_sound():
    audio = pyaudio.PyAudio()

    print( "-------------------------" )
    print( "Let's record some sounds!")
    print( "This script is going to listen to your microphone input" )
    print( "And record tiny audio files to be used for learning later" )
    print( "-------------------------" )
    
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
                print(" - ", dirname[len(glob_path) - 2:-1])
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

    print("You can pause/resume the recording session using the [SPACE] key, and stop the recording using the [ESC] key" )

    global streams
    global recordQueue
    global audios
    global files_recorded
    files_recorded = 0
    streams = {}
    audios = {}
    recordQueue = {
        'status': 'recording'
    }
    labels = [directory]
    if( countdown( 5 ) == False ):
        return;
    
    global currently_recording
    currently_recording = True
    
    time_string = str(int(time.time()))
    for index, microphone_index in enumerate(valid_mics):
        FULL_WAVE_OUTPUT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/source/mici_" + str(microphone_index) + "__" + time_string + ".wav"
        SRT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/segments/mici_" + str(microphone_index) + "__" + time_string + "v" + str(CURRENT_VERSION) + ".srt"
        non_blocking_record(labels, FULL_WAVE_OUTPUT_FILENAME, SRT_FILENAME, microphone_index, index==0)
    
    # wait for stream to finish (5)
    while currently_recording:
        time.sleep(0.1)
    
    for microphone_index in valid_mics:
        streams['index' + str(microphone_index)].stop_stream()
        streams['index' + str(microphone_index)].close()
        audios['index' + str(microphone_index)].terminate()
    
# Consumes the recordings in a sliding window fashion - Always combining the two latest chunks together    
def record_consumer(labels, FULL_WAVE_OUTPUT_FILENAME, SRT_FILE, MICROPHONE_INPUT_INDEX, audio, streams, print_stuff=False):
    global recordQueue
    global currently_recording
    global files_recorded
    indexedQueue = recordQueue['index' + str(MICROPHONE_INPUT_INDEX)]

    amount_of_streams = len(streams)
    detection_strategy = CURRENT_DETECTION_STRATEGY
    
    ms_per_frame = math.floor(RECORD_SECONDS / SLIDING_WINDOW_AMOUNT * 1000)
    detection_labels = []
    for label in labels:
        detection_labels.append(DetectionLabel(label, 0, "", 0, 0, 0, 0))
    detection_state = DetectionState(detection_strategy, "recording", ms_per_frame, 0, True, 0, 0, detection_labels)

    audioFrames = []    
    false_occurrence = []
    current_occurrence = []
    index = 0    
    detection_frames = []
    
    if print_stuff:
        current_status = get_current_status(detection_state)
        for line in current_status:
            print( line )
    
    totalAudioFrames = []
    try:
        with KeyPoller() as key_poller:
            # Write the source file first with the right settings to add the headers, and write the data later
            totalWaveFile = wave.open(FULL_WAVE_OUTPUT_FILENAME, 'wb')
            totalWaveFile.setnchannels(CHANNELS)
            totalWaveFile.setsampwidth(audio.get_sample_size(FORMAT))
            totalWaveFile.setframerate(RATE)
            totalWaveFile.close()
            
            # This is used to modify the wave file directly later
            # Thanks to hydrogen18.com for offering the wav file explanation and code
            CHUNK_SIZE_OFFSET = 4
            DATA_SUB_CHUNK_SIZE_SIZE_OFFSET = 40

            LITTLE_ENDIAN_INT = struct.Struct('<I')
            totalFrameCount = 0
        
            while( currently_recording ):
                while( not indexedQueue.empty() ):
                    audioFrames.append( indexedQueue.get() )
                    detection_state.ms_recorded += ms_per_frame
                    audioFrames, detection_state, detection_frames, current_occurrence, false_occurrence = \
                        process_audio_frame(index, audioFrames, detection_state, detection_frames, current_occurrence, false_occurrence)
                    totalAudioFrames.append( audioFrames[-1] )
                    index += 1
                                                
                    if( record_controls( key_poller, recordQueue ) == False ):
                        for stream in streams:
                            streams[stream].stop_stream()
                        currently_recording = False
                        break;
                         
                    if print_stuff:
                        current_status = get_current_status(detection_state)
                        reset_previous_lines(len(current_status))
                        for line in current_status:
                            print( line )

                    # Append to the total wav file only once every ten audio frames ( roughly once every 225 milliseconds )
                    if (len(totalAudioFrames) >= 15 ):
                        byteString = b''.join(totalAudioFrames)
                        totalFrameCount += len(byteString)
                        totalAudioFrames = []
                        appendTotalFile = open(FULL_WAVE_OUTPUT_FILENAME, 'ab')
                        appendTotalFile.write(byteString)
                        appendTotalFile.close()
                        
                        # Set the amount of frames available and chunk size
                        # By overriding the header part of the wave file manually
                        # Which wouldn't be needed if the wave package supported appending properly                        
                        # Thanks to hydrogen18.com for the explanation and code
                        appendTotalFile = open(FULL_WAVE_OUTPUT_FILENAME, 'r+b')
                        appendTotalFile.seek(0,2)
                        chunk_size = appendTotalFile.tell() - 8
                        appendTotalFile.seek(CHUNK_SIZE_OFFSET)
                        appendTotalFile.write(LITTLE_ENDIAN_INT.pack(chunk_size))
                        appendTotalFile.seek(DATA_SUB_CHUNK_SIZE_SIZE_OFFSET)
                        sample_length = 2 * totalFrameCount
                        appendTotalFile.write(LITTLE_ENDIAN_INT.pack(sample_length))
                        appendTotalFile.close()
                sleep(0.001)
                    
    except Exception as e:
        print( "----------- ERROR DURING RECORDING -------------- " )
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        for stream in streams:
            streams[stream].stop_stream()
        currently_recording = False

def multithreaded_record( in_data, frame_count, time_info, status, queue ):
    queue.put( in_data )
    
    return in_data, pyaudio.paContinue
                
# Records a non blocking audio stream and saves the source and SRT file for it
def non_blocking_record(labels, FULL_WAVE_OUTPUT_FILENAME, SRT_FILE, MICROPHONE_INPUT_INDEX, print_logs):
    global recordQueue
    global streams
    global audios
    
    mic_index = 'index' + str(MICROPHONE_INPUT_INDEX)
    
    recordQueue[mic_index] = Queue(maxsize=0)
    micindexed_lambda = lambda in_data, frame_count, time_info, status, queue=recordQueue['index' + str(MICROPHONE_INPUT_INDEX)]: multithreaded_record(in_data, frame_count, time_info, status, queue) 
    audios[mic_index] = pyaudio.PyAudio()
    streams[mic_index] = audios[mic_index].open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        input_device_index=MICROPHONE_INPUT_INDEX,
        frames_per_buffer=round( RATE * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT ),
        stream_callback=micindexed_lambda)

    consumer = threading.Thread(name='consumer', target=record_consumer, args=(labels, FULL_WAVE_OUTPUT_FILENAME, SRT_FILE, MICROPHONE_INPUT_INDEX, audios[mic_index], streams, print_logs))
    consumer.setDaemon( True )
    consumer.start()
    streams[mic_index].start_stream()
    
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
