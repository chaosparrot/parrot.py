from config.config import *
import pyaudio
import wave
import time
from time import sleep
import scipy.io.wavfile
import audioop
import math
import numpy as np
from numpy.fft import rfft
from scipy.signal import blackmanharris
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pyaudio
from lib.listen import start_nonblocking_listen_loop, predict_wav_files, validate_microphone_input
from lib.machinelearning import feature_engineering
import csv
from lib.audio_model import AudioModel

def test_data( with_intro ):
    available_models = []
    for fileindex, file in enumerate(os.listdir( CLASSIFIER_FOLDER )):
        if ( file.endswith(".pkl") ):
            available_models.append( file )

    available_replays = []
    for fileindex, file in enumerate(os.listdir( REPLAYS_FOLDER )):
        if ( file.endswith(".csv") ):
            available_replays.append( file )
            
    available_sounds = []
    for fileindex, file in enumerate(os.listdir( RECORDINGS_FOLDER )):
        available_sounds.append( file )
            
    if( len( available_models ) == 0 ):
        print( "It looks like you haven't trained any models yet..." )
        print( "Please train an algorithm first using the [L] option in the main menu" )
        return
    elif( with_intro ):
        print("-------------------------")
        print("We can analyze our performance in two different ways")
        print(" - [A] for analyzing an audio stream ( useful for comparing models )")
        print(" - [R] for analyzing an existing replay file")
        print(" - [U] for analyzing a set of recordings for statistical purposes")
        print(" - [M] for analyzing the accuracy of recordings on a specific model")        
        print(" - [X] for exiting analysis mode")
        
    #learn_data = pd.read_csv( 'model_training_bronze_league_1.pkl1578823389.csv', skiprows=0, header=0)
    #for index in range(0,199):
    #    plot_bars( learn_data, index )
    
    analyze_replay_or_audio( available_models, available_replays, available_sounds )
        
def replay( available_replays ):
    if( len( available_replays ) == 1 ):
        replay_index = 0
    else:
        print( "-------------------------" )    
        print( "Select a replay to analyze:" )
        print( "( empty continues the list of replays, [X] exit this mode )" )
        for index, replay in enumerate( available_replays ): 
            filesize = os.path.getsize( REPLAYS_FOLDER + "/" + replay )

            print( "- [" + str( index + 1 ) + "] - " + replay + " (" + str( filesize / 1000 ) + "kb)" )
            replay_prompt = ( index + 1 ) % 10 == 0 or index + 1 == len( available_replays )
            if( replay_prompt ):
                replay_index = input("")
                if( replay_index == "" ):
                    continue
                elif( replay_index == "x" ):
                    return
                else:
                    replay_index = int( replay_index ) - 1
                    break

    while( replay_index == "" ):
        replay_index = input("")
        if( replay_index == "" ):
            continue
        if( replay_index == "x" ):
            return
        else:
            replay_index = int( replay_index ) - 1
                    
    replay_file = available_replays[ replay_index ]
    print( "Analyzing " + replay_file )
    plot_replay( pd.read_csv( REPLAYS_FOLDER + "/" + replay_file, skiprows=0, header=0) )
    
    # Go back to main menu afterwards
    test_data( True )
    
def recording_statistics( available_sounds ):
    print( "-------------------------" )    
    print( " - [X] exit this mode" )
    for sound_index, available_sound in enumerate(available_sounds):
        print( " - [" + str( sound_index + 1 ) + "] " + available_sound )

    print( "Select an audio folder to analyze statistically:" )        
    audio_folder_index = input("")
    if( audio_folder_index.lower() == "x" ):
        test_data( True )
        return   
    while( int( audio_folder_index ) <= 0 ):
        audio_folder_index = input("")

    audio_folder_index = int( audio_folder_index ) - 1
                    
    audio_folder = available_sounds[ audio_folder_index ]
    print( "Analyzing " + audio_folder )
    plot_audio( RECORDINGS_FOLDER + "/" + audio_folder )
    
    # Go back to main menu afterwards
    test_data( True )
    
def plot_audio( folder ):
    wav_files = os.listdir(folder)

    # First sort the wav files by time
    full_wav_files = []
    for wav_file in wav_files:
        if( wav_file.endswith(".wav") ):
            full_wav_files.append( os.path.join( folder, wav_file ) )
            
    intensities = []
    frequencies = []
            
    for index, wav_file in enumerate( full_wav_files ):
        features, frequency = feature_engineering( wav_file, RECORD_SECONDS, FEATURE_ENGINEERING_TYPE )
        intensity = features[ len( features ) - 1 ]
        frequency = features[ len( features ) - 2 ]
        
        frequencies.append( frequency )
        intensities.append( intensity )
        
        print( "Analyzing file " + str( index + 1 ) + " - Intensity %0d - Frequency: %0d           " % ( intensity, frequency) , end="\r")

    print( "                                                                                           ", end="\r" )    
    
    print( "Intensity ------- " )
    print( "Average: " + str( np.average( intensities ) ) )
    print( "Lowest: " + str( min( intensities ) ) )    
    print( "Standard deviation: " + str( np.std( intensities ) ) )
    print( "Frequency ------- " )
    print( "Lowest: " + str( min( frequencies ) ) )    
    print( "Average: " + str( np.average( frequencies ) ) )
    print( "Standard deviation: " + str( np.std( frequencies ) ) )
        
def audio_analysis( available_models ):
    print( "-------------------------" )
    print( "Putting our algorithms to the test!")
    print( "This script is going to load in your model" )
    print( "And test it against audio files" )
    print( "-------------------------" )
    
    classifier = choose_classifier( available_models )
    
    if not os.path.exists(REPLAYS_AUDIO_FOLDER ):
        os.makedirs(REPLAYS_AUDIO_FOLDER)    

    wav_files = os.listdir(REPLAYS_AUDIO_FOLDER)
    
    print( "Should we analyse the existing audio files or record a new set?" )
    print( " - [E] for using the existing files" )
    print( " - [N] for clearing the files and recording new ones" )
    new_or_existing = ""
    while( new_or_existing == "" ):
        new_or_existing = input("")
    
    if( new_or_existing.lower() == "n" ):
        audio = pyaudio.PyAudio()
        if (validate_microphone_input(audio) == False):
            return
    
        for wav_file in wav_files:
            file_path = os.path.join(REPLAYS_AUDIO_FOLDER, wav_file)
            if( file_path.endswith(".wav") ):
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print("Could not delete " + wav_file)
        
        print( "-------------------------" )
        seconds = input("How many seconds of audio should we record? ( default is 15 )" )
        if( seconds == "" ):
            seconds = 15
        else:
            seconds = int( seconds )
            
        # Give the user a delay of 5 seconds before audio recording begins
        for i in range( -5, 0 ):
            print("Recording in... " + str(abs(i)), end="\r")
            sleep( 1 )
            
        print( "Recording new audio files" )
        replay_file = start_nonblocking_listen_loop( classifier, False, True, True, seconds, True )
        classifier = None
        print( "-------------------------" )
        print( "Analyzing file " + replay_file )
        #plot_replay( pd.read_csv( replay_file, skiprows=0, header=0) )
    elif( new_or_existing.lower() == "e" ):
        print( "-------------------------" )
        print( "Analysing existing audio files" )
        full_wav_files = []
        
        # First sort the wav files by time
        raw_wav_filenames = []
        for wav_file in wav_files:
            if( wav_file.endswith(".wav") ):
                raw_wav_filenames.append( float( wav_file.replace(".wav", "") ) )
                
        raw_wav_filenames.sort()
        
        for float_wav_file in raw_wav_filenames:
            wav_file_name = '%0.3f.wav' % ( float_wav_file )
            file_path = os.path.join(REPLAYS_AUDIO_FOLDER, wav_file_name )
            full_wav_files.append( file_path )
                
        predictions = predict_wav_files( classifier, full_wav_files )
        
        dataRows = []
        with open(REPLAYS_FOLDER + '/analysis-replay-' + str(time.time()) + '.csv', 'a', newline='') as csvfile:    
            headers = ['time', 'winner', 'intensity', 'frequency', 'power', 'actions', 'buffer']
            headers.extend( classifier.classes_ )
            if ('silence' not in classifier.classes_):
                headers.extend(['silence'])
            writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
            writer.writeheader()
                    
            for index, prediction in enumerate( predictions ):
                timeString = full_wav_files[ index ].replace( REPLAYS_AUDIO_FOLDER + os.sep, "" ).replace( ".wav", "" )
                dataRow = {'time': int(float(timeString) * 1000) / 1000, 'intensity': 0, 'power': 0, 'actions': [], 'buffer': 0 }
                for column in prediction:
                    dataRow[column] = prediction[ column ]['percent']
                    if( prediction[ column ]['winner'] ):
                        dataRow['winner'] = column
                        dataRow['frequency'] = prediction[column]['frequency']
                        dataRow['intensity'] = prediction[column]['intensity']         
                        dataRow['power'] = prediction[column]['power']

                writer.writerow( dataRow )
                csvfile.flush()
                dataRows.append( dataRow )

        classifier = None
        print( "-------------------------" )
        print( "Analyzing replay!" )
        
        plot_replay( pd.DataFrame(data=dataRows) )
            
    # Go back to main menu afterwards
    test_data( True )
    
def choose_classifier( available_models ):
    if( len( available_models ) == 1 ):
        classifier_file_index = 0
    else:            
        print( "Available models:" )
        for modelindex, available_model in enumerate(available_models):
            print( " - [" + str( modelindex + 1 ) + "] " + available_model )
        classifier_file_index = input("Type the number of the model that you want to test: ")
        if( classifier_file_index == "" ):
            classifier_file_index = 0
        elif( int( classifier_file_index ) > 0 ):
            classifier_file_index = int( classifier_file_index ) - 1
        
    classifier_file = CLASSIFIER_FOLDER + "/" + available_models[ classifier_file_index ]
    print( "Loading model " + classifier_file )
    classifier = joblib.load( classifier_file )
    
    if( not isinstance( classifier, AudioModel ) ):
        settings = {
            'version': 0,
            'RATE': RATE,
            'CHANNELS': CHANNELS,
            'RECORD_SECONDS': RECORD_SECONDS,
            'SLIDING_WINDOW_AMOUNT': SLIDING_WINDOW_AMOUNT,
            'feature_engineering': FEATURE_ENGINEERING_TYPE
        }
        classifier = AudioModel( settings, classifier )

    print( "This model can detect the following classes: " + ", ".join( classifier.classes_ ) )
    
    return classifier

def test_accuracy( available_models, available_sounds ):
    print( "-------------------------" )
    print( "Putting our algorithms to the test!")
    print( "This script is going to load in your model" )
    print( "And test its accuracy against the given sounds" )
    print( "-------------------------" )
    
    classifier = choose_classifier( available_models )
    
    threshold = input("At what percentage should we consider a sound accurately detected? " )
    if( threshold == "" ):
        threshold = 0
    else:
        threshold = int( threshold )
    
    print( "Analysing..." )
    for index, sound in enumerate(available_sounds):
        if( sound in classifier.classes_ ):    
            # First sort the wav files by time
            recordings_dir = os.path.join(RECORDINGS_FOLDER, sound )
            wav_files = os.listdir(recordings_dir)            
            full_wav_files = []
            
            print( "----- " + str(sound) + " -----" )
            i = 0
            for wavindex, wav_file in enumerate(wav_files):
                if( wav_file.endswith(".wav") and i < 5000 ):
                    file_path = os.path.join(recordings_dir, wav_file )
                    full_wav_files.append( file_path )
                    i += 1

            predictions = predict_wav_files( classifier, full_wav_files )
            i_correct = 0
            for prediction in predictions:
                if( sound in prediction and prediction[sound]['winner'] and prediction[sound]['percent'] >= threshold ):
                    i_correct += 1
            print( "Accuracy above threshold %0d - %0.1f " % ( threshold, round((i_correct / i) * 100.0) ) )
        
    test_data(True)
    
def analyze_replay_or_audio( available_models, available_replays, available_sounds ):
    replay_or_audio = input( "" )
    if( replay_or_audio.lower() == "r" ):
        if( len(available_replays ) == 0 ):
            print( "No replays to be analyzed yet - Make sure to do a practice run using the play mode first" )
            analyze_replay_or_audio( available_models, available_replays, available_sounds )
        else:
            replay( available_replays )
    elif( replay_or_audio.lower() == "a" ):
        audio_analysis( available_models )
    elif( replay_or_audio.lower() == "u" ):
        recording_statistics( available_sounds )
    elif( replay_or_audio.lower() == "m" ):
        test_accuracy( available_models, available_sounds )        
    elif( replay_or_audio.lower() == "x" ):
        print("")
        return
    else:
        analyze_replay_or_audio( available_models, available_replays, available_sounds )
        
def plot_replay( replay_data ):
    plt.style.use('seaborn-darkgrid')
    num = 0
    bottom=0
    
    if ('expected' in replay_data.drop([], axis=1)):    
        plt.subplot(2, 1, 1)
        plt.title("Percentage correct vs incorrect", loc='left', fontsize=12, fontweight=0, color='black')
        plt.ylabel("Percentage")
        
        correct_frames = []
        wrong_frames = []
        silence_frames = []
        for index, row in replay_data.iterrows():
            correct = 0
            silence = 0
            wrong = 0
            if (row['winner'] == 'silence'):
                silence = 100
            elif (row['winner'] != 'silence'):
                if (row['expected'] in row):
                    correct = row[row['expected']]
                else:
                    correct = 0
                wrong = 100 - correct
            correct_frames.append(correct)
            wrong_frames.append(wrong)
            silence_frames.append(silence)            
        
        expected_actual_frame = pd.DataFrame({'correct': correct_frames, 'wrong': wrong_frames, 'silence': silence_frames})
        
        for column in expected_actual_frame:
            if( column != "silence" ):
                if ( column == "wrong" ):
                    color = 'red'
                    label = 'Incorrect'
                elif ( column == "correct" ):
                    color = 'green'
                    label = 'Correct'
                num+=1
                plt.bar(np.arange(replay_data['time'].size), expected_actual_frame[column], color=color, linewidth=1, alpha=0.9, label=label, bottom=bottom)
                bottom += np.array( expected_actual_frame[column] )
    else:
        colors = ['darkviolet','red', 'gold', 'green', 'deepskyblue', 'navy', 'gray', 'black', 'pink',
            'firebrick', 'orange', 'lawngreen', 'darkturquoise', 'khaki', 'indigo', 'blue', 'teal',
            'cyan', 'seagreen', 'silver', 'saddlebrown', 'tomato', 'steelblue', 'lavenderblush', 'orangered', 'gray', 'blue', 'red', 'gold', 'pink', 
            'purple', 'indigo', 'khaki', 'darkgray', 'black', 'darkgreen', 'deepskyblue', 'orange',
            'cyan', 'seagreen', 'silver', 'saddlebrown', 'tomato', 'steelblue', 'lavenderblush', 'orangered', 'gray', 'blue', 'red', 'gold', 'pink']
        
        # Add percentage plot
        plt.subplot(2, 1, 1)
        plt.title("Percentage distribution of predicted sounds", loc='left', fontsize=12, fontweight=0, color='black')
        plt.ylabel("Percentage")

        for column in replay_data.drop(['winner', 'intensity', 'time', 'frequency', 'actions', 'buffer', 'power'], axis=1):
            if( column != "silence" ):
                color = colors[num]        
                num+=1
                plt.bar(np.arange(replay_data['time'].size), replay_data[column], color=color, linewidth=1, alpha=0.9, label=column, bottom=bottom)
                bottom += np.array( replay_data[column] )
            
    plt.legend(loc=1, bbox_to_anchor=(1, 1.3), ncol=4)

    ax1 = plt.subplot(2, 1, 2)

    # Add audio subplot
    plt.title('Audio', loc='left', fontsize=12, fontweight=0, color='black')
    ax1.set_ylabel('Loudness', color='green')
    ax1.set_xlabel("Time( in files )")
    ax1.set_ylim(ymax=40000)
    ax1.tick_params('y', colors='black')
    ax1.bar(np.arange(replay_data['time'].size), np.array( replay_data['intensity'] ), color='green', linewidth=1)
    
    frequencyAxis = ax1.twinx()
    frequencyAxis.plot(np.arange(replay_data['time'].size), replay_data['frequency'], '-', color='red')
    frequencyAxis.set_ylabel('Frequency', color='red')
    frequencyAxis.set_ylim(ymax=800)

    plt.show()
    
def plot_bars( learning_data, index ):
    plt.style.use('seaborn-darkgrid')
    plt.figure(num=None, figsize=(20, 10), dpi=80)
    num = 0

    colors = ['darkviolet','red', 'gold', 'green', 'deepskyblue', 'navy', 'gray', 'black', 'pink',
        'firebrick', 'orange', 'lawngreen', 'darkturquoise', 'khaki', 'indigo', 'blue', 'teal',
        'cyan', 'seagreen', 'silver', 'saddlebrown', 'tomato', 'steelblue', 'lavenderblush', 'orangered', 'gray', 'blue', 'red', 'gold', 'pink']

    learning_row = learning_data.iloc[index]
    validation_accuracy = str(int(learning_row['validation_accuracy'] * 1000) / 10)
        
    # Add percentage plot
    string_epoch = str( int(learning_row['epoch'] + 1) )
    plt.title( "Epoch " + string_epoch + " - Validation accuracy " + validation_accuracy + " - Percentage distribution of sounds", loc='left', fontsize=12, fontweight=0, color='black')
    plt.ylabel("Percentage")
    for column in learning_data.drop(['epoch', 'validation_accuracy', 'loss'], axis=1):
        color = colors[num]        
        num+=1
        plt.bar(num, learning_row[column] * 100, color=color, linewidth=1, alpha=0.9, label=column)
            
    plt.legend(loc=1, ncol=7)
    plt.savefig('data/evolution_images/8884-' + string_epoch + '.png')			