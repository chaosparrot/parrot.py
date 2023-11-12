import math
import numpy as np
from scipy.fftpack import fft, rfft, fft2, dct
import audioop
from python_speech_features import mfcc
from .mfsc import Mfsc
from typing import List, Tuple
import os
from config.config import RATE
from scipy import signal

long_byte_size = 4
_mfscs = {}

# Determine the decibel based on full scale of 16 bit ints ( same as Audacity )
def determine_dBFS(waveData: np.array) -> float:
    power = determine_power(waveData)
    if power <= 0:
        power = 0.0001

    return 20 * math.log10(power / math.pow(32767, 2))

def determine_power(waveData: np.array) -> float:
    return audioop.rms(waveData, long_byte_size)

# This power measurement is the old representation for human readability
def determine_legacy_power(waveData: np.array) -> float:
    return determine_power(audioop.rms(waveData, 4)) / 1000

# Old fundamental frequency finder - this one doesn't show frequency in Hz
def determine_legacy_frequency(waveData: np.array) -> float:
    fft_result = fft( waveData )
    positiveFreqs = np.abs( fft_result[ 0:round( len(fft_result) / 2 ) ] )
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

# Approximate vocal formants F1 and F2 using weighted average
# Goal is to have a light weight, smooth pair of values that can be properly controlled by the user
# Heuristics taken based on https://home.cc.umanitoba.ca/~krussll/phonetics/acoustic/formants.html
# 241 taken from assumption 15ms * 16khz + 1
def determine_formant_frequencies(waveData: np.array, bin_size: float = 241) -> Tuple[float, float]:
    bin_range = 8000 / bin_size

    # Check what the loudest frequency is in the 1000Hz range first
    f1_range = int(bin_size / 8)
    # Initially start F2 range from 1100Hz
    f2_range = int(bin_size / 8 + 3)
    
    fft_bins = np.fft.rfft(waveData)
    
    f1_bins = fft_bins[:f1_range]
    f1_n_loudest_bins = 5    
    loudest_f1_fft_bins = np.argpartition(f1_bins, -f1_n_loudest_bins)[-f1_n_loudest_bins:]
    loudest_f1_bin_values = np.take(fft_bins, loudest_f1_fft_bins)
    f1_bin_sum = np.sum(loudest_f1_bin_values)
    f1_weighted_avg = np.real(np.average(loudest_f1_fft_bins, weights=(loudest_f1_bin_values / f1_bin_sum)))
    f1 = max(0, f1_weighted_avg) * bin_range
    
    # Incase the F1 is lower than 600Hz, lower the F2 range start to find low sounding vowels' F2
    if (f1 < 550):
        f2_range = int(bin_size / 8 * 0.8)
    
    f2_bins = fft_bins[f2_range:]
    f2_n_loudest_bins = 20
    loudest_f2_fft_bins = np.argpartition(f2_bins, -f2_n_loudest_bins)[-f2_n_loudest_bins:]
    loudest_f2_bin_values = np.take(f2_bins, loudest_f2_fft_bins)
    f2_bin_sum = np.sum(loudest_f2_bin_values)
    
    # Append the offset of the indexes of f2 to make sure the bins line up with the original fft bins
    f2_weighted_avg = np.real(np.average([f2_bin + f2_range for f2_bin in loudest_f2_fft_bins], weights=(loudest_f2_bin_values / f2_bin_sum)))
    f2 = f2_weighted_avg * bin_range
    
    return f1, f2

def determine_mfcc_type1(waveData: np.array, sampleRate: int = 16000) -> List[float]:
    return mfcc( waveData, samplerate=sampleRate, nfft=1103, numcep=13, appendEnergy=True )
    
def determine_mfcc_type2(waveData: np.array, sampleRate: int = 16000) -> List[float]:
    return mfcc( waveData, samplerate=sampleRate, nfft=1103, numcep=30, nfilt=40, preemph=0.5, winstep=0.005, winlen=0.015, appendEnergy=False )

def determine_mfsc(waveData: np.array, sampleRate:int = 16000) -> List[float]:
    global _mfscs
    if ( sampleRate not in _mfscs ):
        _mfscs[sampleRate] = Mfsc(sr=sampleRate, n_mel=40, preem_coeff=0.5, frame_stride_ms=5, frame_size_ms=15)
    _mfsc = _mfscs[sampleRate]
    return _mfsc.apply( waveData )

# Get a feeling of how much the signal changes based on the total distance between mel frames
def determine_euclidean_dist(mfscData: np.array) -> float:
    mel_frame_amount = len(mfscData)
    distance = 0
    for i in range(0, mel_frame_amount):
        if i > 0:
            distance += np.linalg.norm(mfscData[i-1] - mfscData[i])
    return distance

# High pass filter that filters out most frequencies below voice level
# In order to improve signal to noise ratio
hp_filter = signal.butter(5, 150, 'highpass', fs=RATE, output='sos')
def high_pass_filter(int16_data: np.array) -> np.array:
    global hp_filter
    if hp_filter is not None:
        max_int16_value = 65535
        signal_data = signal.sosfilt(hp_filter, int16_data.astype(np.float32) / max_int16_value) * max_int16_value
        filtered_data = signal_data.astype(np.int16)
        
        return filtered_data
    else:
        return int16_data