from config.config import *
import pyaudio
import struct
import wave
import math
import numpy as np
from lib.print_status import get_current_status
from lib.stream_processing import process_audio_frame, post_processing
from lib.typing import DetectionState, DetectionFrame
from typing import List
import io

class StreamRecorder:
    total_wav_filename: str
    srt_filename: str
    thresholds_filename: str
    comparison_wav_filename: str
    
    audio: pyaudio.PyAudio
    stream: pyaudio.Stream
    detection_state: DetectionState
    
    length_per_frame: int
    audio_frames: List[np.array]
    total_audio_frames: List[np.array]
    index: int
    detection_frames: List[DetectionFrame]
    current_occurrence: List[DetectionFrame]
    false_occurrence: List[DetectionFrame]
    
    def __init__(self, audio: pyaudio.PyAudio, stream: pyaudio.Stream, total_wav_filename: str, srt_filename: str, detection_state: DetectionState):
        self.total_wav_filename = total_wav_filename
        self.srt_filename = srt_filename
        self.comparison_wav_filename = srt_filename.replace(".v" + str(CURRENT_VERSION) + ".srt", "_comparison.wav")
        self.thresholds_filename = srt_filename.replace(".v" + str(CURRENT_VERSION) + ".srt", "_thresholds.txt")
        
        self.audio = audio
        self.stream = stream
        self.detection_state = detection_state
        self.total_audio_frames = []
        self.audio_frames = []
        self.detection_frames = []
        self.current_occurrence = []
        self.false_occurrence = []
        self.index = 0
        self.length_per_frame = 0
    
        # Write the source file first with the right settings to add the headers, and write the data later
        totalWaveFile = wave.open(self.total_wav_filename, 'wb')
        totalWaveFile.setnchannels(CHANNELS)
        totalWaveFile.setsampwidth(audio.get_sample_size(FORMAT))
        totalWaveFile.setframerate(RATE)
        totalWaveFile.close()
    
    # Add a single audio frame to the batch and start processing it
    def add_audio_frame(self, frame: List[np.array]):
        if self.length_per_frame == 0:
            self.length_per_frame = len(frame)
    
        self.index += 1
        self.audio_frames.append(frame)
        self.detection_state.ms_recorded += self.detection_state.ms_per_frame
        audioFrames, detection_state, detection_frames, current_occurrence, false_occurrence = \
            process_audio_frame(self.index, self.audio_frames, self.detection_state, self.detection_frames, self.current_occurrence, self.false_occurrence)
        
        self.current_occurence = current_occurrence
        self.false_occurrence = false_occurrence
        self.detection_state = detection_state
        self.detection_frames = detection_frames
        self.audio_frames = audioFrames
        self.total_audio_frames.append( audioFrames[-1] )
        
        # Append to the total wav file only once every fifteen audio frames 
        # This is roughly once every 225 milliseconds
        if len(self.total_audio_frames) >= 15:
            self.persist_total_wav_file()
    
    def persist_total_wav_file(self):    
        # This is used to modify the wave file directly
        CHUNK_SIZE_OFFSET = 4
        DATA_SUB_CHUNK_SIZE_SIZE_OFFSET = 40
        LITTLE_ENDIAN_INT = struct.Struct('<I')
    
        byteString = b''.join(self.total_audio_frames)
        self.total_audio_frames = []
        appendTotalFile = open(self.total_wav_filename, 'ab')
        appendTotalFile.write(byteString)
        appendTotalFile.close()
        
        # Set the amount of frames available and chunk size
        # By overriding the header part of the wave file manually
        # Which wouldn't be needed if the wave package supported appending properly                        
        # Thanks to hydrogen18.com for the explanation and code
        appendTotalFile = open(self.total_wav_filename, 'r+b')
        appendTotalFile.seek(0,2)
        chunk_size = appendTotalFile.tell() - 8
        appendTotalFile.seek(CHUNK_SIZE_OFFSET)
        appendTotalFile.write(LITTLE_ENDIAN_INT.pack(chunk_size))
        appendTotalFile.seek(DATA_SUB_CHUNK_SIZE_SIZE_OFFSET)
        sample_length = 2 * ( self.index * self.length_per_frame )
        
        appendTotalFile.write(LITTLE_ENDIAN_INT.pack(sample_length))
        appendTotalFile.close()

    # Resume playing
    def resume(self):
        self.stream.start_stream()
    
    def pause(self):
        self.stream.stop_stream()
        
        self.index -= len(self.total_audio_frames)
        if self.index != 0 and len(self.total_audio_frames) > 0:
            self.detection_frames = self.detection_frames[:-len(self.total_audio_frames)]
        self.total_audio_frames = []
        self.audio_frames = []
    
    # Clear out the last N seconds and pauses the stream, returns whether it should resume after
    def clear(self, seconds: float) -> bool:
        should_resume = self.detection_state != "paused"
        self.pause()
        
        ms_per_frame = self.detection_state.ms_per_frame
        frames_to_remove = math.floor(seconds * 1000 / ms_per_frame)
        clear_file = False
        if (self.index < frames_to_remove):
            clear_file = True
        self.index -= self.index if clear_file else frames_to_remove
        self.current_occurrence = []
        self.false_occurrence = []
        
        self.detection_frames = self.detection_frames[:-frames_to_remove]
        self.detection_state.ms_recorded = len(self.detection_frames) * ms_per_frame
        for label in self.detection_state.labels:
            label.ms_detected = 0
            for frame in self.detection_frames:
                if frame.label == label.label:
                    label.ms_detected += ms_per_frame
        
        # Just completely overwrite the file if we go back to the start for simplicities sake
        if clear_file:
            totalWaveFile = wave.open(self.total_wav_filename, 'wb')
            totalWaveFile.setnchannels(CHANNELS)
            totalWaveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
            totalWaveFile.setframerate(RATE)
            totalWaveFile.close()
            
        # Truncate the frames from the total wav file
        else:
            with open(self.total_wav_filename, 'r+b') as f:
                # Drop the last N bytes from the file
                f.seek(-frames_to_remove * self.length_per_frame, io.SEEK_END)
                f.truncate()
                
                # Overwrite the total recording length
                CHUNK_SIZE_OFFSET = 4
                DATA_SUB_CHUNK_SIZE_SIZE_OFFSET = 40
                LITTLE_ENDIAN_INT = struct.Struct('<I')

                f.seek(0,2)
                chunk_size = f.tell() - 8
                f.seek(CHUNK_SIZE_OFFSET)
                f.write(LITTLE_ENDIAN_INT.pack(chunk_size))
                f.seek(DATA_SUB_CHUNK_SIZE_SIZE_OFFSET)
                sample_length = 2 * ( self.index * self.length_per_frame )
                f.write(LITTLE_ENDIAN_INT.pack(sample_length))        
        
        return should_resume
    
    # Reset the counts of the state so they count up nicely during reprocessing of multiple streams
    def reset_label_count(self):
        for label in self.detection_state.labels:
            label.ms_detected = 0
    
    def get_detection_state(self) -> DetectionState:
        return self.detection_state
    
    def get_status(self, detection_states: List[DetectionState] = []) -> List[str]:
        return get_current_status(self.detection_state, detection_states)
    
    # Stop processing the streams and build the final files
    def stop(self, callback = None):
        self.pause()
        self.persist_total_wav_file()
        if self.index == 0:
            os.remove(self.total_wav_filename)        

        comparison_wav_file = wave.open(self.comparison_wav_filename, 'wb')
        comparison_wav_file.setnchannels(1)
        comparison_wav_file.setsampwidth(2)
        comparison_wav_file.setframerate(RATE)
        post_processing(self.detection_frames, self.detection_state, self.srt_filename, self.thresholds_filename, callback, comparison_wav_file)
        self.stream.close()
        self.audio.terminate()
        self.detection_frames = []

    # Do all post processing related tasks that cannot be done during runtime
    def post_processing(self, callback = None, comparison_wav_file: wave.Wave_write = None):
        self.persist_total_wav_file()
        post_processing(self.detection_frames, self.detection_state, self.srt_filename, self.thresholds_filename, callback, comparison_wav_file)
