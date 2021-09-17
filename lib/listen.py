import numpy as np
from config.config import *
from lib.machinelearning import feature_engineering, feature_engineering_raw, get_label_for_directory, get_highest_intensity_of_wav_file, get_recording_power
import pyaudio
import wave
import time
import scipy
import scipy.io.wavfile
import hashlib
import os
import operator
import audioop
import math
import time
import csv
from queue import *
import threading
import traceback
import sys
import lib.ipc_manager as ipc_manager
import joblib
from lib.audio_model import AudioModel
from lib.stream_controls import manage_loop_state
from lib.key_poller import KeyPoller
STATE_POLLING_THRESHOLD = 0.2
    
def classify_audioframes( audioQueue, audio_frames, classifier, high_speed ):
    if( not audioQueue.empty() ):
        audio_frames.append( audioQueue.get() )

        # In case we are dealing with frames not being met and a buffer being built up,
        # Start skipping every other audio frame to maintain being up to date,
        # Trading being up to date over being 100% correct in sequence
        if( audioQueue.qsize() > 1 ):
            print( "SKIP FRAME", audioQueue.qsize() )
            audioQueue.get()
        
        if( len( audio_frames ) >= 2 ):
            audio_frames = audio_frames[-2:]
                        
            highestintensity = np.amax( audioop.maxpp( audio_frames[1], 4 ) / 32767 )
            wavData = b''.join(audio_frames)
                
            # SKIP FEATURE ENGINEERING COMPLETELY WHEN DEALING WITH SILENCE
            if( high_speed == True and highestintensity < SILENCE_INTENSITY_THRESHOLD ):
                probabilityDict, predicted, frequency = create_empty_probability_dict( classifier, {}, 0, highestintensity, 0 )
            else:
                fftData = np.frombuffer( wavData, dtype=np.int16 )
                power = get_recording_power( fftData, classifier.get_setting('RECORD_SECONDS', RECORD_SECONDS) )            
                probabilityDict, predicted, frequency = predict_raw_data( wavData, classifier, highestintensity, power )
            
            return probabilityDict, predicted, audio_frames, highestintensity, frequency, wavData
            
    return False, False, audio_frames, False, False, False
    
def action_consumer( stream, classifier, dataDicts, persist_replay, replay_file, mode_switcher=False ):
    actions = []
    global listening_state
    
    starttime = time.time()
    try:
        if( persist_replay ):
            with open(replay_file, 'a', newline='') as csvfile:    
                headers = ['time', 'winner', 'intensity', 'frequency', 'power', 'actions', 'buffer']
                headers.extend( classifier.classes_ )
                if ('silence' not in classifier.classes_):
                    headers.extend(['silence'])
                writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
                writer.writeheader()
            
                while( listening_state['currently_recording'] == True ):
                    if( not listening_state['classifierQueue'].empty() ):
                        current_time = time.time()
                        seconds_playing = time.time() - starttime
                        
                        listening_state['last_audio_update'] = current_time                    
                        probabilityDict = listening_state['classifierQueue'].get()
                        dataDicts.append( probabilityDict )
                        if( len(dataDicts) > PREDICTION_LENGTH ):
                            dataDicts.pop(0)
                
                        if( mode_switcher ):
                            actions = mode_switcher.getMode().handle_input( dataDicts )
                            if( isinstance( actions, list ) == False ):
                                actions = []
                            
                        replay_row = { 'time': int(seconds_playing * 1000) / 1000, 'actions': ':'.join(actions), 'buffer': listening_state['classifierQueue'].qsize()}
                        for label, labelDict in probabilityDict.items():
                            replay_row[ label ] = labelDict['percent']
                            if( labelDict['winner'] ):
                                replay_row['winner'] = label
                            replay_row['intensity'] = int(labelDict['intensity'])
                            replay_row['power'] = int(labelDict['power'])
                            replay_row['frequency'] = labelDict['frequency']                
                        writer.writerow( replay_row )
                                                
                        csvfile.flush()
                    else:
                        time.sleep( RECORD_SECONDS / 3 )
        else:
            while( listening_state['currently_recording'] == True ):
                if( not listening_state['classifierQueue'].empty() ):
                    current_time = time.time()
                    listening_state['last_audio_update'] = current_time
                    probabilityDict = listening_state['classifierQueue'].get()
                    dataDicts.append( probabilityDict )
                    if( len(dataDicts) > PREDICTION_LENGTH ):
                        dataDicts.pop(0)
            
                    if( mode_switcher ):
                        actions = mode_switcher.getMode().handle_input( dataDicts )
                        if( isinstance( actions, list ) == False ):
                            actions = []
                else:
                    time.sleep( RECORD_SECONDS / 3 )
    except Exception as e:
        print( "----------- ERROR DURING CONSUMING ACTIONS -------------- " )
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        listening_state['stream'].stop_stream()
        listening_state['currently_recording'] = False

    
def classification_consumer( audio, stream, classifier, persist_files, high_speed ):
    audio_frames = []
    dataDicts = []
    for i in range( 0, PREDICTION_LENGTH ):
        dataDict = {}
        for directoryname in classifier.classes_:
            dataDict[ directoryname ] = {'percent': 0, 'intensity': 0, 'frequency': 0, 'winner': False}
        dataDicts.append( dataDict )

    starttime = time.time()
    global listening_state
    
    try:    
        while( listening_state['currently_recording'] == True ):
            probabilityDict, predicted, audio_frames, highestintensity, frequency, wavData = classify_audioframes( listening_state['audioQueue'], audio_frames, classifier, high_speed )
            
            # Skip if a prediction could not be made
            if( probabilityDict == False ):
                time.sleep( classifier.get_setting('RECORD_SECONDS', RECORD_SECONDS) / 3 )
                continue
                
            seconds_playing = time.time() - starttime
                
            winner = classifier.classes_[ predicted ]                
            prediction_time = time.time() - starttime - seconds_playing
            
            #long_comment = "Time: %0.2f - Prediction in: %0.2f - Winner: %s - Percentage: %0d - Frequency %0d                                        " % (seconds_playing, prediction_time, winner, probabilityDict[winner]['percent'], probabilityDict[winner]['frequency'])
            short_comment = "T %0.3f - [%0d%s %s] F:%0d P:%0d" % (seconds_playing, probabilityDict[winner]['percent'], '%', winner, frequency, probabilityDict[winner]['power'])            
            if( winner != "silence" ):
                print( short_comment )
            
            listening_state['classifierQueue'].put( probabilityDict )
            if( persist_files ):        
                audioFile = wave.open(REPLAYS_AUDIO_FOLDER + "/%0.3f.wav" % (seconds_playing), 'wb')
                audioFile.setnchannels(classifier.get_setting('CHANNELS', CHANNELS))
                audioFile.setsampwidth(audio.get_sample_size(FORMAT))
                audioFile.setframerate(classifier.get_setting('RATE', RATE))
                audioFile.writeframes(wavData)
                audioFile.close()
    except Exception as e:
        print( "----------- ERROR DURING AUDIO CLASSIFICATION -------------- " )
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        listening_state['stream'].stop_stream()
        listening_state['currently_recording'] = False
    
    
def nonblocking_record( in_data, frame_count, time_info, status ):
    global listening_state
    listening_state['audioQueue'].put( in_data )
    
    return in_data, pyaudio.paContinue
    
def start_nonblocking_listen_loop( classifier, mode_switcher = False, persist_replay = False, persist_files = False, amount_of_seconds=-1, high_speed=False ):
    global listening_state
    listening_state = {
        'currently_recording': True,
        'stream': None,
        'audioQueue': Queue(maxsize=0),
        'classifierQueue': Queue(maxsize=0),
        'classifier_name': ipc_manager.getClassifier(),
        'restart_listen_loop': False,
        'last_audio_update': time.time()
    }
    
    # Get a minimum of these elements of data dictionaries
    dataDicts = []
    audio_frames = []
    for i in range( 0, PREDICTION_LENGTH ):
        dataDict = {}
        for directoryname in classifier.classes_:
            dataDict[ directoryname ] = {'percent': 0, 'intensity': 0, 'frequency': 0, 'winner': False}
        dataDicts.append( dataDict )
    
    starttime = time.time()
    replay_file = REPLAYS_FOLDER + "/replay_" + str(int(starttime)) + ".csv"
    
    infinite_duration = amount_of_seconds == -1
    audio = pyaudio.PyAudio()
    if ( validate_microphone_input(audio) == False ):
        return None
    
    if( infinite_duration ):
        print( "Listening..." )
    else:
        print ( "Listening for " + str( amount_of_seconds ) + " seconds..." )
    print ( "" )
    
    listening_state['stream'] = audio.open(format=FORMAT, channels=classifier.get_setting('CHANNELS', CHANNELS),
        rate=classifier.get_setting('RATE', RATE), input=True,
        input_device_index=INPUT_DEVICE_INDEX,
        frames_per_buffer=round( classifier.get_setting('RATE', RATE) * classifier.get_setting('RECORD_SECONDS', RECORD_SECONDS) / classifier.get_setting('SLIDING_WINDOW_AMOUNT', SLIDING_WINDOW_AMOUNT) ),
        stream_callback=nonblocking_record)
                
    classificationConsumer = threading.Thread(name='classification_consumer', target=classification_consumer, args=(audio, listening_state['stream'], classifier, persist_files, high_speed) )
    classificationConsumer.setDaemon( True )
    classificationConsumer.start()
    
    actionConsumer = threading.Thread(name='action_consumer', target=action_consumer, args=(listening_state['stream'], classifier, dataDicts, persist_replay, replay_file, mode_switcher) )
    actionConsumer.setDaemon( True )
    actionConsumer.start()
    
    listening_state['last_audio_update'] = time.time()
    listening_state['stream'].start_stream()
    ipc_manager.setParrotState("running")

    while listening_state['currently_recording'] == True and listening_state['restart_listen_loop'] == False:
        currenttime = time.time()
        
        with KeyPoller() as key_poller:
            if( not infinite_duration and currenttime - starttime > amount_of_seconds or manage_loop_state( "running", listening_state, mode_switcher, currenttime, STATE_POLLING_THRESHOLD, key_poller ) == False ):
                listening_state['currently_recording'] = False
        time.sleep(STATE_POLLING_THRESHOLD)

    # Stop all the streams and different threads
    listening_state['stream'].stop_stream()
    listening_state['stream'].close()
    audio.terminate()
    listening_state['audioQueue'].queue.clear()
    listening_state['classifierQueue'].queue.clear()
    classificationConsumer.join()
    actionConsumer.join()
    
    # Restarting the listening loop is required when we are dealing with a different classifier
    # As different classifiers might have different audio requirements
    if (listening_state['restart_listen_loop'] == True):
        classifier = load_running_classifier(ipc_manager.getClassifier())
        listening_state['restart_listen_loop'] = False
        listening_state['currently_recording'] = True
        return start_nonblocking_listen_loop(classifier, mode_switcher, persist_replay, persist_files, amount_of_seconds, high_speed)
    else:
        return replay_file

def predict_wav_files( classifier, wav_files ):
    dataDicts = []
    audio_frames = []
    print ( "Analyzing " + str( len( wav_files) ) + " audio files..." )
    print ( "" )
    
    for i in range( 0, PREDICTION_LENGTH ):
        dataDict = {}
        for directoryname in classifier.classes_:
            dataDict[ directoryname ] = {'percent': 0, 'intensity': 0}
        dataDicts.append( dataDict )

    probabilities = []
    for index, wav_file in enumerate( wav_files ):
        highestintensity = get_highest_intensity_of_wav_file( wav_file, classifier.get_setting('RECORD_SECONDS', RECORD_SECONDS) )
        probabilityDict, predicted, frequency = predict_wav_file( wav_file, classifier, highestintensity )
                
        winner = classifier.classes_[predicted]
        #print( "Analyzing file " + str( index + 1 ) + " - Winner: %s - Percentage: %0d - Frequency: %0d           " % (winner, probabilityDict[winner]['percent'], probabilityDict[winner]['frequency']) , end="\r")
        probabilities.append( probabilityDict )

    print( "                                                                                           ", end="\r" )
    
    return probabilities
        
def predict_raw_data( wavData, classifier, intensity, power ):
    # FEATURE ENGINEERING
    first_channel_data = np.frombuffer( wavData, dtype=np.int16 )
    if( classifier.get_setting('CHANNELS', CHANNELS) == 2 ):
        first_channel_data = first_channel_data[::2]
        
    data_row, frequency = feature_engineering_raw( first_channel_data, classifier.get_setting('RATE', RATE), intensity, classifier.get_setting('RECORD_SECONDS', RECORD_SECONDS), 
        classifier.get_setting('FEATURE_ENGINEERING_TYPE', FEATURE_ENGINEERING_TYPE) )    
    data = [ data_row ]

    return create_probability_dict( classifier, data, frequency, intensity, power )
        
def predict_wav_file( wav_file, classifier, intensity ):
    # FEATURE ENGINEERING
    data_row, frequency = feature_engineering( wav_file, classifier.get_setting('RECORD_SECONDS', RECORD_SECONDS), classifier.get_setting('FEATURE_ENGINEERING_TYPE', FEATURE_ENGINEERING_TYPE) )
    data = [ data_row ]
    
    if( intensity < SILENCE_INTENSITY_THRESHOLD ):
        return create_empty_probability_dict( classifier, data, frequency, intensity, 0 )
    else:
        return create_probability_dict( classifier, data, frequency, intensity, 0 )

def create_empty_probability_dict( classifier, data, frequency, intensity, power ):
    probabilityDict = {}
    index = 0
    predicted = -1
    
    for label in classifier.classes_:
        winner = False
        percent = 0
        if( label == 'silence' ):
            predicted = index
            percent = 100
            winner = True
            
        probabilityDict[ label ] = { 'percent': percent, 'intensity': int(intensity), 'winner': winner, 'frequency': frequency, 'power': power }
        index += 1
        
    if ('silence' not in classifier.classes_):
        probabilityDict['silence'] = { 'percent': 100, 'intensity': int(intensity), 'winner': True, 'frequency': frequency, 'power': power }
            
    return probabilityDict, predicted, frequency
    
def create_probability_dict( classifier, data, frequency, intensity, power ):
    # Predict the outcome of the audio file    
    probabilities = classifier.predict_proba( data ) * 100
    probabilities = probabilities.astype(int)

    # Get the predicted winner        
    predicted = np.argmax( probabilities[0] )
    if( isinstance(predicted, list) ):
        predicted = predicted[0]
    
    probabilityDict = {}
    for index, percent in enumerate( probabilities[0] ):
        label = classifier.classes_[ index ]
        probabilityDict[ label ] = { 'percent': percent, 'intensity': int(intensity), 'winner': index == predicted, 'frequency': frequency, 'power': power }
        
    if ('silence' not in classifier.classes_):
        probabilityDict['silence'] = { 'percent': 100, 'intensity': int(intensity), 'winner': False, 'frequency': frequency, 'power': power }
        
    return probabilityDict, predicted, frequency
    
# Load in a classifier that also sets the classifier state during runtime
def load_running_classifier( classifier_name ):
    if( classifier_name != "dummy" ):
        print( "Loading classifier " + CLASSIFIER_FOLDER + "/" + classifier_name + ".pkl" )
        classifier = joblib.load( CLASSIFIER_FOLDER + "/" + classifier_name + ".pkl" )
        
        if( not isinstance( classifier, AudioModel ) ):
            settings = {
                'version': 0,
                'RATE': RATE,
                'CHANNELS': CHANNELS,
                'RECORD_SECONDS': RECORD_SECONDS,
                'SLIDING_WINDOW_AMOUNT': SLIDING_WINDOW_AMOUNT,
                'FEATURE_ENGINEERING_TYPE': FEATURE_ENGINEERING_TYPE
            }
            
            classifier = AudioModel( settings, classifier )
        ipc_manager.setClassifier(classifier_name)            
    else:
        print( "Loading dummy classifier for testing purposes" )
        from lib.dummy_classifier import DummyClassifier
        classifier = DummyClassifier()
        ipc_manager.setClassifier("dummy")

    return classifier
    
# Validate and print the currently used microphone
def validate_microphone_input( audio ):
    try:
        micDict = audio.get_device_info_by_index( INPUT_DEVICE_INDEX )
        if (micDict and micDict['maxInputChannels'] > 0):
            print( "Using input from " + micDict['name'] )
            return True
        else:
            raise IOError( "Invalid number of channels" )
    except IOError as e:
        print( "------ ERROR - NO VALID MICROPHONE FOUND DURING START UP ------ " )
        print( "Make sure your microphone is connected before starting up Parrot" )
        print( "or change the INPUT_DEVICE_INDEX in the config/config.py file.")
        print( "And rerun Parrot to have the proper connection" )
        print( "---------------------------------------------------------------")
        
    return False

    