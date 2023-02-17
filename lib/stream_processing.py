from .typing import DetectionLabel, DetectionFrame, DetectionEvent, DetectionState
from config.config import BACKGROUND_LABEL, RECORD_SECONDS, SLIDING_WINDOW_AMOUNT, RATE
from typing import List
import wave
import math
import numpy as np
from .signal_processing import determine_power, determine_dBFS, determine_mfsc, determine_euclidean_dist
from .srt import persist_srt_file, print_detection_performance_compared_to_srt
import os
import audioop

# When converting to ints from bytes, Windows uses a 32 bit number.
# Other OSes use the bytes shown. So for Windows we need different calculations for frame count
# ( https://stackoverflow.com/questions/72482769/numpy-returns-different-results-on-windows-and-unix )
long_byte_size = 4 if os.name == 'nt' else 2

CURRENT_VERSION = 1

def process_wav_file(input_file, srt_file, output_file, labels, progress_callback = None, comparison_srt_file = None, print_statistics = False):
    audioFrames = []
    edgesAudioFrames = []
    wf = wave.open(input_file, 'rb')
    number_channels = wf.getnchannels()
    total_frames = wf.getnframes()
    frame_rate = wf.getframerate()
    frames_to_read = round( frame_rate * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT )
    ms_per_frame = math.floor(RECORD_SECONDS / SLIDING_WINDOW_AMOUNT * 1000)
    sample_width = 2# 16 bit = 2 bytes
    
    detection_strategy = "auto_dBFS_mend_dBFS_30ms_secondary_dBFS_reject_cont_45ms_repair"
    
    detection_labels = []
    for label in labels:
        detection_labels.append(DetectionLabel(label, 0, "", 0, 0, 0, 0))
    detection_state = DetectionState(detection_strategy, "recording", ms_per_frame, 0, True, 0, 0, detection_labels)    

    false_occurrence = []
    current_occurrence = []
    index = 0    
    detection_frames = []

    if progress_callback is not None:
        progress_callback(0, detection_state)    
    
    while( wf.tell() < total_frames ):
        index = index + 1
        raw_wav = wf.readframes(frames_to_read * number_channels)
        detection_state.ms_recorded += ms_per_frame
        detected = False        
        
        # If our wav file is shorter than the amount of bytes ( assuming 16 bit ) times the frames, we discard it and assume we arrived at the end of the file
        if (len(raw_wav) != 2 * frames_to_read * number_channels ):
            break;        
        
        # Do online downsampling if the files frame rate is higher than our 16k Hz rate
        # To make sure all the calculations stay accurate
        if frame_rate > RATE:
            raw_wav, _ = audioop.ratecv(raw_wav, sample_width, number_channels, frame_rate, RATE, None)
            if number_channels > 1:
                raw_wav = audioop.tomono(raw_wav[0], 2, 1, 0)

        audioFrames.append(raw_wav)
        if( len( audioFrames ) >= SLIDING_WINDOW_AMOUNT ):
            audioFrames = audioFrames[-SLIDING_WINDOW_AMOUNT:]
            
            byteString = b''.join(audioFrames)
            wave_data = np.frombuffer( byteString, dtype=np.int16 )
            power = determine_power( wave_data )
            dBFS = determine_dBFS( wave_data )
            mfsc_data = determine_mfsc( wave_data, RATE )
            distance = determine_euclidean_dist( mfsc_data )                

            # Attempt to detect a label
            detected_label = BACKGROUND_LABEL
            for label in detection_state.labels:
                if is_detected(detection_state.strategy, power, dBFS, distance, label.min_dBFS):
                    detected = True
                    label.ms_detected += ms_per_frame
                    detected_label = label.label
                    break

            detection_frames.append(DetectionFrame(index, ms_per_frame, detected, power, dBFS, distance, mfsc_data, detected_label))
            if detected:
                current_occurrence.append(detection_frames[-1])
            else:
                false_occurrence.append(detection_frames[-1])
        else:
            detection_frames.append(DetectionFrame(index, ms_per_frame, False, 0, 0, 0, [], BACKGROUND_LABEL))
            false_occurrence.append(detection_frames[-1])                
            
        # Recalculate the noise floor / signal strength every 10 frames
        # For performance reason and because the statistical likelyhood of things changing every 150ms is pretty low
        if len(detection_frames) % 10 == 0:
            detection_state = determine_detection_state(detection_frames, detection_state)            

        # On-line rejection - This may be undone in post-processing later
        # Only add occurrences longer than 75 ms as no sound a human produces is shorter
        if detected == False and len(current_occurrence) > 0:
            is_continuous = False
            for label in detection_state.labels:
                if label == current_occurrence[0].label:
                    is_continuous = label.duration_type == "continuous"
                    break

            if is_rejected(detection_state.strategy, current_occurrence, detection_state.ms_per_frame, is_continuous):
                total_rejected_frames = len(current_occurrence)
                for frame_index in range(-total_rejected_frames - 1, 0, 1):
                    rejected_frame_index = frame_index
                    detection_frames[rejected_frame_index].label = BACKGROUND_LABEL
                    detection_frames[rejected_frame_index].positive = False
            current_occurrence = []
        # On-line mending - This may be undone in post-processing later
        # Only keep false detections longer than a certain amount ( because a human can't make them shorter )
        elif detected and len(false_occurrence) > 0:            
            if is_mended(detection_state.strategy, false_occurrence, detection_state, detected_label):
                total_mended_frames = len(false_occurrence)
                for frame_index in range(-total_mended_frames - 1, 0, 1):
                    mended_frame_index = frame_index
                    detection_frames[mended_frame_index].label = detected_label
                    detection_frames[mended_frame_index].positive = True                
            false_occurrence = []
        
        # Convert from different byte sizes to 16bit for proper progress
        progress = wf.tell() / total_frames
        if progress_callback is not None and progress < 1:
            # For the initial pass we calculate 75% of the progress
            # This progress partitioning is completely arbitrary
            progress_callback(progress * 0.75, detection_state)

    wf.close()
    
    output_wave_file = wave.open(output_file, 'wb')
    output_wave_file.setnchannels(number_channels)
    output_wave_file.setsampwidth(sample_width)
    output_wave_file.setframerate(RATE)
    
    post_processing(detection_frames, detection_state, srt_file, progress_callback, output_wave_file, comparison_srt_file, print_statistics )
    progress = 1
    if progress_callback is not None:
        progress_callback(progress, detection_state)    

def post_processing(frames: List[DetectionFrame], detection_state: DetectionState, output_filename: str, progress_callback = None, output_wave_file: wave.Wave_write = None, comparison_srt_file: str = None, print_statistics = False) -> List[DetectionFrame]:
    detection_state.state = "processing"
    if progress_callback is not None:
        progress_callback(0, detection_state)
    
    # Do a full pass on all the frames again to fix labels we might have missed
    if "repair" in detection_state.strategy:    
        current_occurrence = []
        false_occurrence = []
        current_label = None
        detected_label = None
        
        # Recalculate the MS detection and duratoin type
        for label in detection_state.labels:
            label.ms_detected = 0
            label.duration_type = determine_duration_type(label, frames)            

        for index, frame in enumerate(frames):
            detected = False
            for label in detection_state.labels:
                if is_detected(detection_state.strategy, frame.power, frame.dBFS, frame.euclid_dist, label.min_dBFS):
                    detected = True
                    label.ms_detected += detection_state.ms_per_frame
                    current_label = label
                    break
        
            # Do a secondary pass if the previous label was negative
            # As we can use its thresholds for correcting late starts
            mending_offset = 0
            if detected and not frames[index - 1].positive:
                 for label in detection_state.labels:
                     if current_label.label == label.label and is_detected_secondary(detection_state.strategy, frames[index - 1].power, frames[index - 1].dBFS, frames[index - 1].euclid_dist, label.min_dBFS - 4):
                         label.ms_detected += detection_state.ms_per_frame
                         frames[index - 1].label = current_label.label
                         frames[index - 1].positive = True
                         mending_offset = -1
                         if len(false_occurrence) > 0:
                             false_occurrence.pop()
                         
                         # Only do two frames of late start fixing as longer late starts statistically do not seem to occur
                         if not frames[index - 2].positive and is_detected_secondary(detection_state.strategy, frames[index - 2].power, frames[index - 2].dBFS, frames[index - 2].euclid_dist, label.min_dBFS - 4):
                             label.ms_detected += detection_state.ms_per_frame
                             frames[index - 2].label = current_label.label
                             frames[index - 2].positive = True
                             mending_offset = -2
                             if len(false_occurrence) > 0:
                                 false_occurrence.pop()
                         break
        
            if detected:
                current_occurrence.append(frame)
                frame.label = current_label.label
                frame.positive = True
                frames[index] = frame
                
                if len(false_occurrence) > 0:                
                    if is_mended(detection_state.strategy, false_occurrence, detection_state, current_label.label):
                        total_mended_frames = len(false_occurrence)
                        current_label.ms_detected += total_mended_frames * detection_state.ms_per_frame                        
                        for frame_index in range(-total_mended_frames - 1 + mending_offset, mending_offset, 1):
                            mended_frame_index = index + frame_index
                            frames[mended_frame_index].label = current_label.label
                            frames[mended_frame_index].positive = True                
                    false_occurrence = []

            if not detected:
                false_occurrence.append(frame)
                frame.positive = False
                frame.label = BACKGROUND_LABEL
                frames[index] = frame
                
                if len(current_occurrence) > 0:
                    is_continuous = False
                    for label in detection_state.labels:
                        if label == current_occurrence[0].label:
                            is_continuous = label.duration_type == "continuous"
                            break
                                    
                    if is_rejected(detection_state.strategy, current_occurrence, detection_state.ms_per_frame, is_continuous):
                        total_rejected_frames = len(current_occurrence)
                        current_label.ms_detected -= total_rejected_frames * detection_state.ms_per_frame
                        current_label = None
                        for frame_index in range(-total_rejected_frames - 1, 0, 1):
                            rejected_frame_index = index + frame_index
                            frames[rejected_frame_index].label = BACKGROUND_LABEL
                            frames[rejected_frame_index].positive = False
                    current_occurrence = []
            
                progress = index / len(frames) 
                if progress_callback is not None and progress < 1:
                    # For the post processing phase - we count the remaining 25% of the progress
                    # This progress partitioning is completely arbitrary
                    progress_callback(0.75 + ( progress * 0.25 ), detection_state)


    # Persist the SRT file
    events = detection_frames_to_events(frames)
    persist_srt_file( output_filename, events )
    
    comparisonOutputWaveFile = None
    if print_statistics:
        if output_wave_file is not None:
            comparisonOutputWaveFile = wave.open(output_filename + "_comparison.wav", 'wb')
            comparisonOutputWaveFile.setnchannels(output_wave_file.getnchannels())
            comparisonOutputWaveFile.setsampwidth(output_wave_file.getsampwidth())
            comparisonOutputWaveFile.setframerate(output_wave_file.getframerate())
        
        print_detection_performance_compared_to_srt(frames, detection_state.ms_per_frame, comparison_srt_file, comparisonOutputWaveFile)

    # Persist the detection wave file
    if output_wave_file is not None:
        frames_to_write = round( RATE * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT )
        sample_width = 2# 16 bit = 2 bytes
        detection_audio_frames = []
        for frame in frames:
            highest_amp = 65536 / 10
            signal_strength = highest_amp if frame.positive else 0

            detection_signal = np.full(int(frames_to_write / sample_width), int(signal_strength))
            detection_signal[::2] = 0
            detection_signal[::3] = 0
            detection_signal[::5] = 0
            detection_signal[::7] = 0
            detection_signal[::9] = 0
            detection_audio_frames.append( detection_signal )    
        output_wave_file.writeframes(b''.join(detection_audio_frames))
        output_wave_file.close()

    detection_state.state = "recording"
    return frames

def determine_detection_state(detection_frames: List[DetectionFrame], detection_state: DetectionState) -> DetectionState:
    # Filter out very low power dbFS values as we can assume the hardware microphone is off
    # And we do not want to skew the mean for that as it would create more false positives
    # ( -70 dbFS was selected as a cut off after a bit of testing with a HyperX Quadcast microphone )
    dBFS_frames = [x.dBFS for x in detection_frames]    
    std_dbFS = np.std(dBFS_frames)
    detection_state.expected_snr = math.floor(std_dbFS * 2)
    detection_state.expected_noise_floor = np.min(dBFS_frames) + std_dbFS
    for label in detection_state.labels:

        # Recalculate the duration type every 15 seconds
        if label.duration_type == "" or len(detection_frames) % round(15 / RECORD_SECONDS):
            label.duration_type = determine_duration_type(label, detection_frames)
        label.min_dBFS = detection_state.expected_noise_floor + detection_state.expected_snr
    return detection_state

# Approximately determine whether the label in the stream is discrete or continuous
# Discrete sounds are from a single source event like a click, tap or a snap
# Whereas continuous sounds have a steady stream of energy from a source
def determine_duration_type(label: DetectionLabel, detection_frames: List[DetectionFrame]) -> str:
    label_events = [x for x in detection_frames_to_events(detection_frames) if x.label == label.label]
    if len(label_events) < 10:
        return ""
    else:
        # The assumption here is that discrete sounds cannot vary in length much as you cannot elongate the sound of a click for example
        # So if the length doesn't vary much, we assume discrete over continuous
        lengths = [x.end_ms - x.start_ms for x in label_events]
        continuous_length_threshold = detection_frames[0].duration_ms * SLIDING_WINDOW_AMOUNT
        return "discrete" if np.std(lengths) < continuous_length_threshold else "continuous"

def detection_frames_to_events(detection_frames: List[DetectionFrame]) -> List[DetectionEvent]:
    events = []
    current_label = ""
    current_frames = []
    for frame in detection_frames:
        if frame.label != current_label:
            if len(current_frames) > 0:
                events.append( DetectionEvent(current_label, current_frames[0].index, current_frames[-1].index, \
                    (current_frames[0].index - 1) * current_frames[0].duration_ms, (current_frames[-1].index) * current_frames[-1].duration_ms, current_frames) )
                current_frames = []
            current_label = frame.label

        if current_label != BACKGROUND_LABEL:
            current_frames.append( frame )
            
    if len(current_frames) > 0:
        events.append( DetectionEvent(current_label, current_frames[0].index, current_frames[-1].index, \
            (current_frames[0].index - 1) * current_frames[0].duration_ms, (current_frames[-1].index) * current_frames[-1].duration_ms, current_frames) )
        current_frames = []
    return events
    
def auto_decibel_detection(power, dBFS, distance, dBFS_threshold):
    return dBFS > dBFS_threshold
    
def auto_secondary_decibel_detection(power, dBFS, distance, dBFS_threshold):
    return dBFS > dBFS_threshold - 7

def is_detected(strategy, power, dBFS, distance, estimated_threshold):
    if "auto_dBFS" in strategy:
       return auto_decibel_detection(power, dBFS, distance, estimated_threshold)

def is_rejected( strategy, occurrence, ms_per_frame, continuous = False ):
    if "reject" not in strategy:
        return False
    elif "reject_45ms" in strategy:
        return len(occurrence) * ms_per_frame < 45
    elif "reject_60ms" in strategy:
        return len(occurrence) * ms_per_frame < 60
    elif "reject_75ms" in strategy:
        return len(occurrence) * ms_per_frame < 75
    elif "reject_90ms" in strategy:
        return len(occurrence) * ms_per_frame < 90
    elif "reject_cont_45ms" in strategy:
        return len(occurrence) * ms_per_frame < ( 45 if continuous else 0 )

def is_detected_secondary( strategy, power, dBFS, distance, estimated_threshold ):
    if "secondary" not in strategy:
        return False
    elif "secondary_dBFS" in strategy:
        return auto_secondary_decibel_detection(power, dBFS, distance, estimated_threshold)

def is_mended( strategy, occurrence, detection_state, current_label ):
    if "mend" not in strategy:
        return False
    elif "mend_60ms" in strategy:
        return len(occurrence) * detection_state.ms_per_frame < 60
    elif "mend_45ms" in strategy:
        return len(occurrence) * detection_state.ms_per_frame < 45
    elif "mend_dBFS" in strategy:
        label_dBFS_threshold = 0
        for label in detection_state.labels:
            if label.label == current_label:
                label_dBFS_threshold = label.min_dBFS
        
        total_missed_length_ms = 0        
        for frame in occurrence:
            if not auto_secondary_decibel_detection(frame.power, frame.dBFS, frame.euclid_dist, label_dBFS_threshold):
                if not "mend_dBFS_30ms" in strategy:
                    return False
                else:
                    total_missed_length_ms += detection_state.ms_per_frame
        if not "mend_dBFS_30ms" in strategy:
            return True
        else:
            return total_missed_length_ms < 30
