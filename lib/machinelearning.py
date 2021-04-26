import hashlib
import scipy
import scipy.io.wavfile
from scipy.fftpack import fft, rfft, fft2, dct
from python_speech_features import mfcc
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from config.config import *
import wave
import audioop
from lib.mfsc import Mfsc
import torchaudio
import torch
if (PYTORCH_AVAILABLE == True):
    from audiomentations import Compose, AddGaussianNoise, Shift, TimeStretch

_mfscs = {}
_mfscsn = {}

def feature_engineering( wavFile, record_seconds, input_type ):
    fs, rawWav = scipy.io.wavfile.read( wavFile )
    intensity = get_highest_intensity_of_wav_file( wavFile, record_seconds )
    
    with wave.open( wavFile ) as fd:
        number_channels = fd.getnchannels()
        if( number_channels == 1 ):
            return feature_engineering_raw( rawWav, fs, intensity, record_seconds, input_type )
        else:
            return feature_engineering_raw( rawWav[:,0], fs, intensity, record_seconds, input_type )        
    
def feature_engineering_raw( wavData, sampleRate, intensity, record_seconds, input_type ):
    freq = get_loudest_freq( wavData, record_seconds )
    if (input_type == TYPE_FEATURE_ENGINEERING_RAW_WAVE):
        data_row = wavData
    elif(input_type == TYPE_FEATURE_ENGINEERING_OLD_MFCC):
        mfcc_result1 = mfcc( wavData, samplerate=sampleRate, nfft=1103, numcep=13, appendEnergy=True )
        data_row = []
        data_row.extend( mfcc_result1.ravel() )
        data_row.append( freq )
        data_row.append( intensity )
    elif(input_type == TYPE_FEATURE_ENGINEERING_NORM_MFCC):
        mfcc_result1 = mfcc( wavData, samplerate=sampleRate, nfft=1103, numcep=30, nfilt=40, preemph=0.5, winstep=0.005, winlen=0.015, appendEnergy=False )
        data_row = []
        data_row.extend( mfcc_result1.ravel() )
    elif(input_type == TYPE_FEATURE_ENGINEERING_NORM_MFSC):
        global _mfscs
        if ( sampleRate not in _mfscs ):
            _mfscs[sampleRate] = Mfsc(sr=sampleRate, n_mel=40, preem_coeff=0.5, frame_stride_ms=5, frame_size_ms=15)

        _mfsc = _mfscs[sampleRate]
        mfsc_result = _mfsc.apply( wavData )
        data_row = []
        data_row.extend( mfsc_result.ravel() )
    elif(input_type == TYPE_FEATURE_ENGINEERING_NOOV_MFSC):
        global _mfscsn
        if ( sampleRate not in _mfscsn ):
            _mfscsn[sampleRate] = Mfsc(sr=sampleRate, n_mel=40, preem_coeff=0.5, frame_stride_ms=10, frame_size_ms=10)

        _mfsc = _mfscsn[sampleRate]
        mfsc_result = _mfsc.apply( wavData )
        data_row = []        
        data_row.extend( mfsc_result.ravel() )
        
    return data_row, freq
    
def training_feature_engineering( wavFile, settings):
    fs, rawWav = scipy.io.wavfile.read( wavFile )
    wavData = rawWav
    if ( settings['CHANNELS'] == 2 ):
        wavData = rawWav[:,0]

    data_row = []
    input_type = settings['FEATURE_ENGINEERING_TYPE']    
    if ( input_type == TYPE_FEATURE_ENGINEERING_NORM_MFCC ):
        mfcc_result1 = mfcc( wavData, samplerate=fs, nfft=1103, numcep=30, nfilt=40, preemph=0.5, winstep=0.005, winlen=0.015, appendEnergy=False )
        data_row.extend( mfcc_result1.ravel() )
    elif ( input_type == TYPE_FEATURE_ENGINEERING_RAW_WAVE ):
        data_row = wavData
    elif(input_type == TYPE_FEATURE_ENGINEERING_NORM_MFSC):
        global _mfscs
        if ( fs not in _mfscs ):
            _mfscs[fs] = Mfsc(sr=fs, n_mel=40, preem_coeff=0.5, frame_stride_ms=5, frame_size_ms=15)

        _mfsc = _mfscs[fs]
        mfsc_result = _mfsc.apply( wavData )
        data_row.extend( mfsc_result.ravel() )
    elif(input_type == TYPE_FEATURE_ENGINEERING_NOOV_MFSC):
        global _mfscsn
        if ( fs not in _mfscsn ):
            _mfscsn[fs] = Mfsc(sr=fs, n_mel=40, preem_coeff=0.5, frame_stride_ms=10, frame_size_ms=10)

        _mfsc = _mfscsn[fs]
        mfsc_result = _mfsc.apply( wavData )
        data_row.extend( mfsc_result.ravel() )

    else:
        print( "OLD MFCC TYPE IS NOT SUPPORTED FOR TRAINING PYTORCH" )
        
    return data_row

def augmented_feature_engineering( wavFile, settings ):
    fs, rawWav = scipy.io.wavfile.read( wavFile )
    wavData = rawWav
    if ( settings['CHANNELS'] == 2 ):
        wavData = rawWav[:,0]
    
    augmenter = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    wavData = augmenter(samples=np.array(wavData, dtype="float32"), sample_rate=fs)
    
    data_row = []
    input_type = settings['FEATURE_ENGINEERING_TYPE']
    if ( input_type == TYPE_FEATURE_ENGINEERING_NORM_MFCC ):
        mfcc_result1 = mfcc( wavData, samplerate=fs, nfft=1103, numcep=30, nfilt=40, preemph=0.5, winstep=0.005, winlen=0.015, appendEnergy=False )
        data_row.extend( mfcc_result1.ravel() )
    elif ( input_type == TYPE_FEATURE_ENGINEERING_RAW_WAVE ):
        data_row = wavData
    elif(input_type == TYPE_FEATURE_ENGINEERING_NORM_MFSC):
        global _mfscs
        if ( fs not in _mfscs ):
            _mfscs[fs] = Mfsc(sr=fs, n_mel=40, preem_coeff=0.5, frame_stride_ms=5, frame_size_ms=15)

        _mfsc = _mfscs[fs]
        mfsc_result = _mfsc.apply( wavData )
        data_row.extend( mfsc_result.ravel() )
    elif(input_type == TYPE_FEATURE_ENGINEERING_NOOV_MFSC):
        global _mfscsn
        if ( fs not in _mfscsn ):
            _mfscsn[fs] = Mfsc(sr=fs, n_mel=40, preem_coeff=0.5, frame_stride_ms=10, frame_size_ms=10)

        _mfsc = _mfscsn[fs]
        mfsc_result = _mfsc.apply( wavData )
        data_row.extend( mfsc_result.ravel() )

    else:
        print( "OLD MFCC TYPE IS NOT SUPPORTED FOR TRAINING PYTORCH" )    
    return data_row

    
def get_label_for_directory( setdir ):
    return float( int(hashlib.sha256( setdir.encode('utf-8')).hexdigest(), 16) % 10**8 )

def cross_validation( classifier, dataset, labels):
    return cross_val_score(classifier, dataset, labels, cv=3)

def average_prediction_speed( classifier, dataset_x ):
    start_time = time.time() * 1000
    classifier.predict( dataset_x[-1000:] )
    end_time = time.time() * 1000
    return int( end_time - start_time ) / len(dataset_x)
    
def create_confusion_matrix(classifier, dataset_x, dataset_labels, all_labels):
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_labels, random_state=1)
    y_pred = classifier.predict( X_test )

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, all_labels )
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True category')
    plt.xlabel('Predicted category')
    plt.show()
    
def get_highest_intensity_of_wav_file( wav_file, record_seconds ):
    intensity = []
    with wave.open( wav_file ) as fd:
        number_channels = fd.getnchannels()
        total_frames = fd.getnframes()
        frame_rate = fd.getframerate()
        frames_to_read = round( frame_rate * record_seconds)
        data = fd.readframes(frames_to_read)
        peak = audioop.maxpp( data, 4 ) / 32767
        intensity.append( peak )
    
    return np.amax( intensity )
        
def get_loudest_freq( wavData, recordLength ):
    fft_result = fft( wavData )

    positiveFreqs = np.abs( fft_result[ 0:round( len(fft_result)/2 ) ] )
    highestFreq = 0
    loudestPeak = 500
    frequencies = [0]
    for freq in range( 0, len( positiveFreqs ) ):
        if( positiveFreqs[ freq ] > loudestPeak ):
            loudestPeak = positiveFreqs[ freq ]
            highestFreq = freq
    
    if( loudestPeak > 500 ):
        frequencies.append( highestFreq )
    
    if( recordLength < 1 ):
        # Considering our sound sample is, for example, 100 ms, our lowest frequency we can find is 10Hz ( I think )
        # So add that as a base to our found frequency to get Hz - This is probably wrong
        freqInHz = ( 1 / recordLength ) + np.amax( frequencies )
    else:
        # I have no clue how to even pretend to know how to calculate Hz for fft frames longer than a second
        freqInHz = np.amax( frequencies )
        
    return freqInHz

def get_recording_power( fftData, recordLength ):
    return audioop.rms( fftData, 4 ) / 1000