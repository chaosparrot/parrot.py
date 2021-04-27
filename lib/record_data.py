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
    global stream
    ESCAPEKEY = '\x1b'
    SPACEBAR = ' '
    
    character = key_poller.poll()
    if(character is not None):    
        if( character == SPACEBAR ):
            print( "Recording paused!" )

            if (stream != None):
                stream.stop_stream()

            # Pause the recording by looping until we get a new keypress
            while( True ):
                
                ## If the audio queue exists - make sure to clear it continuously
                if( recordQueue != None ):
                    recordQueue.queue.clear()
            
                character = key_poller.poll()
                if(character is not None):                
                    if( character == SPACEBAR ):
                        print( "Recording resumed!" )
                        if (stream is not None):
                            stream.start_stream()
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
    if (validate_microphone_input(audio) == False):
        return;

    print( "-------------------------" )
    print( "Let's record some sounds!")
    print( "This script is going to listen to your microphone input" )
    print( "And record tiny audio files to be used for learning later" )
    print( "-------------------------" )

    directory = input("Whats the name of the sound are you recording? ")
    while (directory == ""):
        directory = input("")
    
    if not os.path.exists(RECORDINGS_FOLDER + "/" + directory):
        os.makedirs(RECORDINGS_FOLDER + "/"  + directory)
    if not os.path.exists(RECORDINGS_FOLDER + "/" + directory + "/source"):
        os.makedirs(RECORDINGS_FOLDER + "/"  + directory + "/source")

    threshold = 0
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

    FULL_WAVE_OUTPUT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/source/i_" + str(threshold) + "__p_" + str(power_threshold) + \
    "__f_" + str(frequency_threshold) + "__begin_" + str(begin_threshold) + "__" + str(int(time.time() ) ) + ".wav"
    WAVE_OUTPUT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/" + str(int(time.time() ) ) + "file";
    WAVE_OUTPUT_FILE_EXTENSION = ".wav";

    global stream
    stream = None
    if( countdown( 5 ) == False ):
        return;
    
    global currently_recording
    currently_recording = True
    non_blocking_record(threshold, power_threshold, frequency_threshold, begin_threshold, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION, FULL_WAVE_OUTPUT_FILENAME)
    
# Consumes the recordings in a sliding window fashion - Always combining the two latest chunks together    
def record_consumer(threshold, power_threshold, frequency_threshold, begin_threshold, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION, FULL_WAVE_OUTPUT_FILENAME, audio, stream):
    global recordQueue

    files_recorded = 0
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
            while( True ):
                if( not recordQueue.empty() ):
                    audioFrames.append( recordQueue.get() )
                    totalAudioFrames.append( audioFrames[-1] )
                    if( len( audioFrames ) >= SLIDING_WINDOW_AMOUNT ):
                        j+=1
                        audioFrames = audioFrames[-SLIDING_WINDOW_AMOUNT:]
                            
                        intensity = [
                            audioop.maxpp( audioFrames[0], 4 ) / 32767,
                            audioop.maxpp( audioFrames[1], 4 ) / 32767
                        ]
                        highestintensity = np.amax( intensity )
                        
                        byteString = b''.join(audioFrames)
                        fftData = np.frombuffer( byteString, dtype=np.int16 )
                        frequency = get_loudest_freq( fftData, RECORD_SECONDS )
                        power = get_recording_power( fftData, RECORD_SECONDS )
                        
                        fileid = "%0.2f" % ((j) * RECORD_SECONDS )
                    
                        if( record_controls( key_poller, recordQueue ) == False ):
                            stream.stop_stream()
                            break;
                             
                        if( frequency > frequency_threshold and highestintensity > threshold and power > power_threshold ):
                            record_wave_file_count += 1
                            if( record_wave_file_count <= begin_threshold and record_wave_file_count > delay_threshold ):
                                files_recorded += 1
                                print( "Files recorded: %0d - Power: %0d - Freq: %0d - Saving %s" % ( files_recorded, power, frequency, fileid ) )
                                waveFile = wave.open(WAVE_OUTPUT_FILENAME + fileid + WAVE_OUTPUT_FILE_EXTENSION, 'wb')
                                waveFile.setnchannels(CHANNELS)
                                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                                waveFile.setframerate(RATE)
                                waveFile.writeframes(byteString)
                                waveFile.close()
                            else:
                                print( "Files recorded: %0d - Power: %0d - Freq: %0d" % ( files_recorded, power, frequency ) )
                        else:
                            record_wave_file_count = 0
                            print( "Files recorded: %0d - Power: %0d - Freq: %0d" % ( files_recorded, power, frequency ) )
                            
                        # Persist the total wave only once every six frames
                        if (len(totalAudioFrames) % 6 ):
                            byteString = b''.join(totalAudioFrames)
                            waveFile = wave.open(FULL_WAVE_OUTPUT_FILENAME, 'wb')
                            waveFile.setnchannels(CHANNELS)
                            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                            waveFile.setframerate(RATE)
                            waveFile.writeframes(byteString)
                            waveFile.close()
                    
                        
    except Exception as e:
        print( "----------- ERROR DURING RECORDING -------------- " )
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        stream.stop_stream()

def multithreaded_record( in_data, frame_count, time_info, status ):
    global recordQueue
    recordQueue.put( in_data )
    
    return in_data, pyaudio.paContinue
                
# Records a non blocking audio stream and saves the chunks onto a queue
# The queue will be used as a sliding window over the audio, where two chunks are combined into one audio file
def non_blocking_record(threshold, power_threshold, frequency_threshold, begin_threshold, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION, FULL_WAVE_OUTPUT_FILENAME):
    global currently_recording
    global recordQueue
    global stream
    recordQueue = Queue(maxsize=0)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        input_device_index=INPUT_DEVICE_INDEX,
        frames_per_buffer=round( RATE * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT ),
        stream_callback=multithreaded_record)
        
    consumer = threading.Thread(name='consumer', target=record_consumer, args=(threshold, power_threshold, frequency_threshold, begin_threshold, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION, FULL_WAVE_OUTPUT_FILENAME, audio, stream))
    consumer.setDaemon( True )
    consumer.start()
    stream.start_stream()

    # wait for stream to finish (5)
    while currently_recording:
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    audio.terminate()