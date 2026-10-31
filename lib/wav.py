import wave
from config.config import BACKGROUND_LABEL, RECORD_SECONDS, SLIDING_WINDOW_AMOUNT, MS_PER_FRAME, RATE, TYPE_FEATURE_ENGINEERING_NORM_MFSC, PYTORCH_AVAILABLE
from .srt import parse_srt_file
import numpy as np
import audioop
from typing import List, Tuple
import os
import struct
import time

CHUNK_SIZE_OFFSET = 4
DATA_SUB_CHUNK_SIZE_SIZE_OFFSET = 40
RIFF_PREFIX_SIZE = 8
WAV_HEADER_SIZE = DATA_SUB_CHUNK_SIZE_SIZE_OFFSET + 4
LITTLE_ENDIAN_INT = struct.Struct('<I')
INT16_FULL_SCALE = 32768.0

# Resamples the audio down to 16kHz ( or any other RATE filled in )
# To make sure all the other calculations are stable and correct
def resample_audio(wavData: np.array, frame_rate, number_channels) -> np.array:
    if frame_rate > RATE:
        sample_width = 2# 16 bit = 2 bytes
        wavData, _ = audioop.ratecv(wavData, sample_width, number_channels, frame_rate, RATE, None)
        if number_channels > 1:
            wavData = audioop.tomono(wavData[0], 2, 1, 0)
    return wavData

def load_wav_files_with_srts( directories, label, int_label, start, end, input_type ):
    category_dataset_x = []
    category_dataset_labels = []
    totalFeatureEngineeringTime = 0
    category_file_index = 0

    for directory in directories:
        source_directory = os.path.join( directory, "source" )
        segments_directory = os.path.join( directory, "segments" )
        
        srt_files = []
        
        for fileindex, file in enumerate(os.listdir(segments_directory)):
            if file.endswith(".srt"):
                srt_files.append(file)
        
        for source_index, source_file in enumerate(os.listdir(source_directory)):
            if source_file.endswith(".wav"):
                full_filename = os.path.join(source_directory, source_file)
                print( "Loading " + str(category_file_index) + " files for " + label + "... ", end="\r" )
                category_file_index += 1
                
                # Find the SRT files available for this source file
                shared_key = source_file.replace(".wav", "")
                possible_srt_files = [x for x in srt_files if x.startswith(shared_key)]
                if len(possible_srt_files) == 0:
                    continue
                
                # Find the highest version of the segmentation for this source file
                srt_file = possible_srt_files[0]
                for possible_srt_file in possible_srt_files:
                    current_version = int( srt_file.replace(".srt", "").replace(shared_key + ".v", "") )
                    version = int( possible_srt_file.replace(".srt", "").replace(shared_key + ".v", "") )
                    if version > current_version:
                        srt_file = possible_srt_file
                full_srt_filename = os.path.join(segments_directory, srt_file)

                # Load the WAV file and turn it into a onedimensional array of numbers
                feature_engineering_start = time.time() * 1000
                data = load_wav_data_from_srt(full_srt_filename, full_filename, input_type, False)
                category_dataset_x.extend( data )
                category_dataset_labels.extend([ label for data_row in data ])
                totalFeatureEngineeringTime += time.time() * 1000 - feature_engineering_start

        print( "Loaded " + str( len( category_dataset_labels ) ) + " .wav files for category " + label + " (id: " + str(int_label) + ")" )
    return category_dataset_x, category_dataset_labels, totalFeatureEngineeringTime

def augment_wav_data(wavData, sample_rate):
    # Imported here to keep lib.wav light to import
    from audiomentations import Compose, AddGaussianNoise, Shift, TimeStretch
    augmenter = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        Shift(p=0.5),
    ])
    return augmenter(samples=np.array(wavData, dtype="float32"), sample_rate=sample_rate)

def load_wav_data_from_srt(srt_file: str, source_file: str, feature_engineering_type = TYPE_FEATURE_ENGINEERING_NORM_MFSC, with_offset = True, should_augment=False, background=False) -> List[List[float]]:
    # Imported here to keep lib.wav light to import
    from lib.machinelearning import feature_engineering_raw
    wav_file_data = []
    wf = wave.open(source_file, 'rb')
    frame_rate = wf.getframerate()
    number_channels = wf.getnchannels()
    total_frames = get_frame_count(wf, source_file)
    frames_to_read = round( frame_rate * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT )    
    ms_per_frame = MS_PER_FRAME
    
    # If offsets are required - We seek half a frame behind the expected frame to get more data from a different location
    halfframe_offset = round( frames_to_read * number_channels * 0.5 )
    start_offsets = [0, -halfframe_offset] if with_offset else [0]
    
    transition_events = parse_srt_file(srt_file, ms_per_frame)
    if len(transition_events) < 2:
        print( "Empty .SRT file for " + source_file + " - Consider deleting " + srt_file + " to resegment the audio file" )
        
    for index, transition_event in enumerate(transition_events):
        next_event_index = total_frames / frames_to_read if index + 1 >= len(transition_events) else transition_events[index + 1].start_index
        audioFrames = []
        
        if (transition_event.label == BACKGROUND_LABEL) == background:
            
            for offset in start_offsets:
                # Skip of the offset makes the position before the start of the file
                if offset + (frames_to_read * transition_event.start_index) < 0:
                    continue;
                wf.setpos(offset + (frames_to_read * transition_event.start_index))
            
                keep_collecting = True
                while keep_collecting:
                    try:
                        raw_wav = wf.readframes(frames_to_read * number_channels)
                    except RuntimeError:
                        raw_wav = ""
                        print( "Error loading in all of the .SRT file for " + source_file + " - Consider deleting " + srt_file + " to resegment the audio file" )
                        keep_collecting = False

                    # Reached the end of wav - do not keep collecting
                    if (len(raw_wav) != SLIDING_WINDOW_AMOUNT * frames_to_read * number_channels ):
                        keep_collecting = False
                        break
                        
                    raw_wav = resample_audio(raw_wav, frame_rate, number_channels)                    
                    audioFrames.append(raw_wav)
                    if( len( audioFrames ) >= SLIDING_WINDOW_AMOUNT ):
                        audioFrames = audioFrames[-SLIDING_WINDOW_AMOUNT:]
                
                        byteString = b''.join(audioFrames)
                        wave_data = np.frombuffer( byteString, dtype=np.int16 )
                        if should_augment and PYTORCH_AVAILABLE:
                            wave_data = augment_wav_data(wave_data, RATE)
                        wav_file_data.append( feature_engineering_raw(wave_data, RATE, 0, RECORD_SECONDS, feature_engineering_type)[0] )
                    
                        if wf.tell() >= ( next_event_index * frames_to_read ) + offset:
                            keep_collecting = False
    
    return wav_file_data

def get_frame_count(wf, path) -> int:
    """Count frames, guarding against an inaccurate header

    Returns:
        frame count, one sample per channel each. e.g. 16000 for 1 s mono at 16 kHz
    """
    frame_bytes = wf.getnchannels() * wf.getsampwidth()
    data_bytes = os.path.getsize(path) - WAV_HEADER_SIZE
    return min(wf.getnframes(), data_bytes // frame_bytes)

def rewrite_wav_sizes(wav_file):
    wav_file.seek(0, 2)
    file_size = wav_file.tell()

    riff_chunk_size = file_size - RIFF_PREFIX_SIZE
    wav_file.seek(CHUNK_SIZE_OFFSET)
    wav_file.write(LITTLE_ENDIAN_INT.pack(riff_chunk_size))

    data_chunk_size = file_size - WAV_HEADER_SIZE
    wav_file.seek(DATA_SUB_CHUNK_SIZE_SIZE_OFFSET)
    wav_file.write(LITTLE_ENDIAN_INT.pack(data_chunk_size))

class UnreadableWav(Exception):
    CANT_OPEN = "cant_open"
    NOT_16_BIT = "not_16_bit"
    NOT_PCM = "not_pcm"
    DAMAGED = "damaged"
    EMPTY = "empty"

    def __init__(self, reason, bits=None):
        super().__init__(reason)
        self.reason = reason
        self.bits = bits

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a whole wav as mono floats. Raises UnreadableWav with the reason

    Returns (samples, rate):
        samples: mono float32 array, -1 to 1 e.g. [0.0021, -0.0154, ...]
        rate: sample rate in Hz e.g. 16000
    """
    try:
        with wave.open(path, "rb") as wf:
            bits = wf.getsampwidth() * 8
            if bits != 16:
                raise UnreadableWav(UnreadableWav.NOT_16_BIT, bits)
            channels = wf.getnchannels()
            rate = wf.getframerate()
            raw = wf.readframes(get_frame_count(wf, path))
    except UnreadableWav:
        raise
    except OSError:
        raise UnreadableWav(UnreadableWav.CANT_OPEN)
    except wave.Error as exc:
        # "unknown format: 3" for float, "unknown extended format" for
        # WAVE_FORMAT_EXTENSIBLE ones, which ffmpeg writes above 16 bits
        not_pcm = str(exc).startswith("unknown")
        raise UnreadableWav(UnreadableWav.NOT_PCM if not_pcm else UnreadableWav.DAMAGED)
    except Exception:
        raise UnreadableWav(UnreadableWav.DAMAGED)
    if not rate:
        raise UnreadableWav(UnreadableWav.DAMAGED)

    # A file cut off mid-sample ends the read there. get_frame_count's cap
    # assumes no chunk before the data, so it doesn't always catch this.
    raw = raw[:len(raw) - len(raw) % (2 * channels)]
    if not raw:
        raise UnreadableWav(UnreadableWav.EMPTY)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        # Interleaved [L0, R0, L1, R1] -> rows [[L0, R0], [L1, R1]] -> mono [avg0, avg1]
        data = data.reshape(-1, channels).mean(axis=1)
    samples = data / INT16_FULL_SCALE
    return samples, rate

def wav_duration(path: str) -> float:
    """Returns duration in seconds. 0.0 if unreadable"""
    try:
        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            return get_frame_count(wf, path) / rate if rate else 0.0
    except Exception:
        return 0.0
