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
from queue import *
import threading
import traceback
import sys
from lib.listen import validate_microphone_input
from lib.key_poller import KeyPoller

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
    ESCAPEKEY = '\x1b'
    SPACEBAR = ' '
    
    character = key_poller.poll()
    if(character is not None):    
        if( character == SPACEBAR ):
            print( "Recording paused!" )

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
                if(character is not None):                
                    if( character == SPACEBAR ):
                        print( "Recording resumed!" )
                        if (streams is not None):
                            for stream in streams:
                                streams[stream].start_stream()
                        return True
                    elif( character == ESCAPEKEY ):
                        print( "Recording stopped" )
                        currently_recording = False
                        return False
                time.sleep(0.3)
        elif( character == ESCAPEKEY ):
            print( "Recording stopped" )
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

    directory = input("Whats the name of the sound are you recording? ")
    while (directory == ""):
        directory = input("")
    
    if not os.path.exists(RECORDINGS_FOLDER + "/" + directory):
        os.makedirs(RECORDINGS_FOLDER + "/"  + directory)
    if not os.path.exists(RECORDINGS_FOLDER + "/" + directory + "/source"):
        os.makedirs(RECORDINGS_FOLDER + "/"  + directory + "/source")

    power_threshold = input("What signal power ( loudness ) threshold do you need? " )
    if( power_threshold == "" ):
        power_threshold = 0
    else:
        power_threshold = int( power_threshold )
        
    frequency_threshold = input("What frequency threshold do you need? " )
    if( frequency_threshold == "" ):
        frequency_threshold = 0
    else:
        frequency_threshold = int( frequency_threshold )
    begin_threshold = 10000
    
    print("")
    print("You can pause/resume the recording session using the [SPACE] key, and stop the recording using the [ESC] key" )

    global streams
    global recordQueue
    global audios
    global files_recorded
    files_recorded = 0
    streams = {}
    audios = {}
    recordQueue = {}
    if( countdown( 5 ) == False ):
        return;
    
    global currently_recording
    currently_recording = True
    
    time_string = str(int(time.time()))
    for index, microphone_index in enumerate(valid_mics):
        FULL_WAVE_OUTPUT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/source/i_0__p_" + str(power_threshold) + \
        "__f_" + str(frequency_threshold) + "__begin_" + str(begin_threshold) + "__mici_" + str(microphone_index) + "__" + time_string + ".wav"
        WAVE_OUTPUT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/" + time_string + "__mici_" + str(microphone_index) + "__file";
        WAVE_OUTPUT_FILE_EXTENSION = ".wav";
        
        non_blocking_record(power_threshold, frequency_threshold, begin_threshold, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION, FULL_WAVE_OUTPUT_FILENAME, microphone_index, index==0)
    
    # wait for stream to finish (5)
    while currently_recording:
        time.sleep(0.1)
    
    for microphone_index in valid_mics:
        streams['index' + str(microphone_index)].stop_stream()
        streams['index' + str(microphone_index)].close()
        audios['index' + str(microphone_index)].terminate()
    
# Consumes the recordings in a sliding window fashion - Always combining the two latest chunks together    
def record_consumer(power_threshold, frequency_threshold, begin_threshold, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION, FULL_WAVE_OUTPUT_FILENAME, MICROPHONE_INPUT_INDEX, audio, streams, print_stuff=False):
    global recordQueue
    global currently_recording
    global files_recorded
    indexedQueue = recordQueue['index' + str(MICROPHONE_INPUT_INDEX)]

    j = 0
    record_wave_file_count = 0
    audioFrames = []
    
    # Set the proper thresholds for starting recordings
    delay_threshold = 0
    if( begin_threshold < 0 ):
        delay_threshold = begin_threshold * -1
        begin_threshold = 1000
    
    totalAudioFrames = []
    try:
        with KeyPoller() as key_poller:
            # Write the source file first with the right settings to add the headers, and write the data later
            totalWaveFile = wave.open(FULL_WAVE_OUTPUT_FILENAME, 'wb')
            totalWaveFile.setnchannels(CHANNELS)
            totalWaveFile.setsampwidth(audio.get_sample_size(FORMAT))
            totalWaveFile.setframerate(RATE)
            totalWaveFile.close()
        
            while( currently_recording ):
                while( not indexedQueue.empty() ):
                    audioFrames.append( indexedQueue.get() )
                    totalAudioFrames.append( audioFrames[-1] )
                    if( len( audioFrames ) >= SLIDING_WINDOW_AMOUNT ):
                        j+=1
                        audioFrames = audioFrames[-SLIDING_WINDOW_AMOUNT:]
                                                
                        byteString = b''.join(audioFrames)
                        fftData = np.frombuffer( byteString, dtype=np.int16 )
                        frequency = get_loudest_freq( fftData, RECORD_SECONDS ) if frequency_threshold > 0 else 0
                        power = get_recording_power( fftData, RECORD_SECONDS )
                        
                        fileid = "%0.2f" % ((j) * RECORD_SECONDS )
                    
                        if( record_controls( key_poller, recordQueue ) == False ):
                            for stream in streams:
                                streams[stream].stop_stream()
                            currently_recording = False
                            break;
                             
                        if( frequency > frequency_threshold and power > power_threshold ):
                            record_wave_file_count += 1
                            if( record_wave_file_count <= begin_threshold and record_wave_file_count > delay_threshold ):
                                files_recorded += 1
                                if print_stuff:
                                    print( "Files recorded: %0d - Power: %0d - Freq: %0d - Saving %s" % ( files_recorded, power, frequency, fileid ) )
                                waveFile = wave.open(WAVE_OUTPUT_FILENAME + fileid + WAVE_OUTPUT_FILE_EXTENSION, 'wb')
                                waveFile.setnchannels(CHANNELS)
                                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                                waveFile.setframerate(RATE)
                                waveFile.writeframes(byteString)
                                waveFile.close()
                            else:
                                if print_stuff:
                                    print( "Files recorded: %0d - Power: %0d - Freq: %0d" % ( files_recorded, power, frequency ) )
                        else:
                            record_wave_file_count = 0
                            if print_stuff:
                                print( "Files recorded: %0d - Power: %0d - Freq: %0d" % ( files_recorded, power, frequency ) )
                            
                    # Append to the total wav file only once every ten audio frames ( roughly once every 225 milliseconds )
                    if (len(totalAudioFrames) >= 15 ):
                        byteString = b''.join(totalAudioFrames)
                        totalAudioFrames = []
                        waveFile = open(FULL_WAVE_OUTPUT_FILENAME, 'ab')
                        waveFile.write(byteString)
                        waveFile.close()
                sleep(0.001)
                    
    except Exception as e:
        print( "----------- ERROR DURING RECORDING -------------- " )
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        stream.stop_stream()

def multithreaded_record( in_data, frame_count, time_info, status, queue ):
    queue.put( in_data )
    
    return in_data, pyaudio.paContinue
                
# Records a non blocking audio stream and saves the chunks onto a queue
# The queue will be used as a sliding window over the audio, where two chunks are combined into one audio file
def non_blocking_record(power_threshold, frequency_threshold, begin_threshold, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION, FULL_WAVE_OUTPUT_FILENAME, MICROPHONE_INPUT_INDEX, print_logs):
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
        
    consumer = threading.Thread(name='consumer', target=record_consumer, args=(power_threshold, frequency_threshold, begin_threshold, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION, FULL_WAVE_OUTPUT_FILENAME, MICROPHONE_INPUT_INDEX, audios[mic_index], streams, print_logs))
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
