from config.config import *
import os
import subprocess
import math
import time
import wave
from queue import *
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.signal import blackmanharris
from lib.machinelearning import get_loudest_freq, get_recording_power
import audioop

def convert_files( with_intro ):            
    available_sounds = []
    for fileindex, file in enumerate(os.listdir( RECORDINGS_FOLDER )):
        if ( os.path.isdir(os.path.join(RECORDINGS_FOLDER, file)) ):
            available_sounds.append( file )
         
    try:
        ffmpeg_configured = True
        subprocess.call([PATH_TO_FFMPEG], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except: 
        ffmpeg_configured = False
            
    if( len( available_sounds ) == 0 ):
        print( "It looks like you haven't recorded any sound yet..." )
        print( "Please make sure to put your audio files inside a subfolder of " + RECORDINGS_FOLDER )
        return
    elif( with_intro ):
        print("-------------------------")
        print("File conversion possible on the following sounds: ")
        print( ", ".join(available_sounds))
        print("-------------------------")        
        if ( ffmpeg_configured ):
            print(" - [W] for transforming .flac to .wav.")
            print( "   Wav files are required to do training or segmentation on." )
            print( "   Audio will be resampled automatically " )
            print(" - [R] for resampling .wav files ( in case you changed the CHANNELS or RATE after recording audio )")
            print(" - [F] for transforming .wav to .flac.")
        else:
            print( "")
            print( "!! FFMPEG was not found at the configured path '" + PATH_TO_FFMPEG + "'. If you desire to convert file extensions install it and point the path to the program at PATH_TO_FFMPEG in the config" )
        print(" - [S] for segmenting a directory of sound files into chunks trainable by Parrot.py")
        print(" - [C] for subclassifying existing wav files using a classifier to provide advanced pruning options.")        
        print(" - [X] to exit conversion mode")
            
    convert_or_segment_files( available_sounds, ffmpeg_configured )
        
def convert_or_segment_files( available_sounds, ffmpeg_configured ):
    convert_or_segment = input( "" )
    if( convert_or_segment.lower() == "w" and ffmpeg_configured ):
        sounds = determine_sounds( available_sounds, "to convert to .wav" )
        convert_audiofile_extension( sounds, ".flac", ".wav", "flac_to_wav", True)
    elif( convert_or_segment.lower() == "f" and ffmpeg_configured ):
        sounds = determine_sounds( available_sounds, "to convert to .flac" )
        convert_audiofile_extension( sounds, ".wav", ".flac", "wav_to_flac" )
    elif( convert_or_segment.lower() == "r" and ffmpeg_configured ):
        sounds = determine_sounds( available_sounds, "to resample .wav to " + str(RATE) + " with " + str(CHANNELS) + " channels" )
        convert_audiofile_extension( sounds, ".wav", ".wav", "resampling_rate_" + str(RATE) + "_channels_" + str(CHANNELS), True)
    elif( convert_or_segment.lower() == "s" ):    
        sounds = determine_sounds( available_sounds, "to segment into " + str(int(RECORD_SECONDS * 1000)) + " chunks")
        segment_audiofiles( sounds )
    elif( convert_or_segment.lower() == "c" ):
        print( "TODO!" )
    elif( convert_or_segment.lower() == "x" ):
        print("")
        return
    else:
        convert_or_segment_files( available_sounds, ffmpeg_configured )

def convert_audiofile_extension( sound_directories, input_extension, output_extension, operation, resampling=False ):
    operation_directory = CONVERSION_OUTPUT_FOLDER + "/" + str(int(time.time())) + "_" + operation
    
    for index, directory in enumerate(sound_directories):
        print( "Converting " + directory + "..." )
        full_directory_path = RECORDINGS_FOLDER + "/" + directory
        output_directory = operation_directory + "/" + directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        files_to_convert = []
        for fileindex, file in enumerate(os.listdir(full_directory_path)):
            if ( file.endswith(input_extension) ):
                files_to_convert.append( file )

        amount_files_to_convert = len(files_to_convert)
        if ( amount_files_to_convert == 0 ):
            print( "No " + input_extension + " files found to convert - skipping" )
        else:
            for convert_index, file in enumerate(files_to_convert):
                input_file = full_directory_path + "/" + file
                output_file = output_directory + "/" + file.replace(input_extension, output_extension)
                if ( resampling == True ):
                    process_to_call = [PATH_TO_FFMPEG, "-i", input_file, "-ar", str(RATE), "-ac", str(CHANNELS), output_file ]
                else:
                    process_to_call = [PATH_TO_FFMPEG, "-i", input_file, output_file ]                
                subprocess.call(process_to_call, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print( str( math.floor(((convert_index + 1 ) / amount_files_to_convert ) * 100)) + "%", end="\r" )
                
            print( "Placed output of conversion in " + output_directory )

def segment_audiofiles(sound_directories):
    operation_directory = CONVERSION_OUTPUT_FOLDER + "/" + str(int(time.time())) + "_segmentation" 
    
    print( "-------------------------" )
    print( "Set up the filtering levels to filter out silence or other unrelated sounds")
    print( "-------------------------" )

    threshold = input("What intensity( loudness ) threshold do you need? " )
    if( threshold == "" ):
        threshold = 0
    else:
        threshold = int(threshold)
        
    power_threshold = input("What signal power threshold do you need? " )
    if( power_threshold == "" ):
        power_threshold = 0
    else:
        power_threshold = int( power_threshold )
        
    frequency_threshold = input("What frequency threshold do you need? " )
    if( frequency_threshold == "" ):
        frequency_threshold = 0
    else:
        frequency_threshold = int( frequency_threshold )
    print( "During a wave of recognized sounds... " )
    begin_threshold = input("After how many saved files should we stop assuming the sound is being made? " )
    if( begin_threshold == "" ):
        begin_threshold = 1000
    else:
        begin_threshold = int( begin_threshold )

    if( begin_threshold == 1000 ):
        begin_threshold = input("After how many positive recognitions should we save the files? " )
        if( begin_threshold == "" ):
            begin_threshold = 1000
        else:
            begin_threshold = 0 - int( begin_threshold )
        
    print("")
    print("You can pause/resume the recording session using the [SPACE] key, and stop the recording using the [ESC] key" )

    for index, directory in enumerate(sound_directories):
        print( "Segmenting " + directory + "..." )
        full_directory_path = RECORDINGS_FOLDER + "/" + directory        
        output_directory = operation_directory + "/" + directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        files_to_segment = []
        for fileindex, file in enumerate(os.listdir(full_directory_path)):
            if ( file.endswith(".wav") ):
                files_to_segment.append( file )

        amount_files_to_segment = len(files_to_segment)
        if ( amount_files_to_segment == 0 ):
            print("No .wav files found to segment - skipping" )
        else:
            for fileindex, file in enumerate(files_to_segment):
                segment_input_file( threshold, power_threshold, frequency_threshold, begin_threshold, full_directory_path + "/" + file, output_directory + "/" + file.replace(".wav", "-"), ".wav" )

# Segments an existing wav file and saves the chunks onto a queue
# The queue will be used as a sliding window over the audio, where two chunks are combined into one audio file
def segment_input_file(threshold, power_threshold, frequency_threshold, begin_threshold, WAVE_INPUT_FILE, WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILE_EXTENSION):
    audioFrames = []

    wf = wave.open(WAVE_INPUT_FILE, 'rb')
    number_channels = wf.getnchannels()
    total_frames = wf.getnframes()
    frame_rate = wf.getframerate()
    frames_to_read = round( frame_rate * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT )
    
    files_recorded = 0
    delay_threshold = 0
    if( begin_threshold < 0 ):
        delay_threshold = begin_threshold * -1
        begin_threshold = 1000

    audio = pyaudio.PyAudio()
    record_wave_file_count = 0
    index = 0
    while( wf.tell() < total_frames ):
        index = index + 1
        raw_wav = wf.readframes(frames_to_read * number_channels)
        
        # If our wav file is shorter than the amount of bytes ( assuming 16 bit ) times the frames, we discard it and assume we arriveed at the end of the file
        if (len(raw_wav) != 2 * frames_to_read * number_channels ):
            break;
        else:
            audioFrames.append(raw_wav)
            if( len( audioFrames ) >= SLIDING_WINDOW_AMOUNT ):
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

                print( "Segmenting file " + WAVE_INPUT_FILE + ": " + str( math.ceil(wf.tell() / total_frames * 100) ) + "%" , end="\r" )
                if( frequency > frequency_threshold and highestintensity > threshold and power > power_threshold ):
                    record_wave_file_count += 1
                    if( record_wave_file_count <= begin_threshold and record_wave_file_count > delay_threshold ):
                        files_recorded += 1
                        waveFile = wave.open(WAVE_OUTPUT_FILENAME + str(index) + WAVE_OUTPUT_FILE_EXTENSION, 'wb')
                        waveFile.setnchannels(number_channels)
                        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                        waveFile.setframerate(frame_rate)
                        waveFile.writeframes(byteString)
                        waveFile.close()
                else:
                    record_wave_file_count = 0
        
    print( "Extracted " + str(files_recorded) + " segmented files from " + WAVE_INPUT_FILE )
    wf.close()

def determine_sounds( available_sounds, verb = "to process" ):
    print( "Selecting sounds " + verb + "... ( [Y]es / [N]o / [S]kip )" )
    filtered_sounds = []
    for sound in available_sounds:
        add = input(" - " + sound)
        if( add == "" or add.strip().lower() == "y" ):
            filtered_sounds.append( sound )
        elif( add.strip().lower() == "s" ):
            break
        else:
            print( "Disabled " + sound )
            
    return filtered_sounds
