from .typing import DetectionLabel, DetectionFrame, DetectionEvent, DetectionState
from config.config import BACKGROUND_LABEL, RECORD_SECONDS, SLIDING_WINDOW_AMOUNT, RATE, CURRENT_VERSION, CURRENT_DETECTION_STRATEGY
from typing import List
import wave
import math
import numpy as np
from .signal_processing import determine_power, determine_dBFS, determine_mfsc, determine_mfsc_shape, determine_euclidean_dist, high_pass_filter, determine_zero_crossing_count
from .wav import resample_audio
from .srt import persist_srt_file, print_detection_performance_compared_to_srt
import os

snr_cutoff = 30

def process_wav_file(input_file, srt_file, output_file, labels, progress_callback = None, comparison_srt_file = None, override_file = None, print_statistics = False):
    audioFrames = []
    wf = wave.open(input_file, 'rb')
    number_channels = wf.getnchannels()
    total_frames = wf.getnframes()
    frame_rate = wf.getframerate()
    frames_to_read = round( frame_rate * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT )
    ms_per_frame = math.floor(RECORD_SECONDS / SLIDING_WINDOW_AMOUNT * 1000)
    sample_width = 2# 16 bit = 2 bytes
    
    detection_strategy = CURRENT_DETECTION_STRATEGY
    
    detection_labels = []
    override_labels = []
    for label in labels:
        detection_labels.append(DetectionLabel(label, 0, 0, "", 0, 0, 0, 0, 0))
    detection_state = DetectionState(detection_strategy, "recording", ms_per_frame, 0, True, 0, 0, 0, 0, detection_labels)
    
    # Add manual overrides if the override file exists
    if override_file is not None and os.path.exists(override_file):
        override_dict = generate_override_dict(override_file)
    
        for override_label in labels:
            duration_type = ""
            duration_type_key = (label + "_duration_type").lower()
            if duration_type_key in override_dict and override_dict[duration_type_key].lower() in ["discrete", "continuous"]:
                duration_type = override_dict[duration_type_key].lower()
                
            min_dBFS = -96
            min_dBFS_key = (label + "_min_dbfs").lower()
            if min_dBFS_key in override_dict and override_dict[min_dBFS_key] < 0:
                min_dBFS = override_dict[min_dBFS_key]
            
            override_labels.append(DetectionLabel(label, 0, 0, duration_type, 0, min_dBFS, 0, 0, 0))    
    detection_state.override_labels = override_labels

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
        raw_wav = resample_audio(raw_wav, frame_rate, number_channels)

        audioFrames.append(raw_wav)
        audioFrames, detection_state, detection_frames, current_occurrence, false_occurrence = \
            process_audio_frame(index, audioFrames, detection_state, detection_frames, current_occurrence, false_occurrence)
        
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

def process_audio_frame(index, audioFrames, detection_state, detection_frames, current_occurrence, false_occurrence):
    detection_frames.append(determine_detection_frame(index, detection_state, audioFrames, detection_frames))
    detected = detection_frames[-1].positive
    previously_detected = len(detection_frames) > 1 and detection_frames[-2].positive
    
    # Keep a base threshold of the current sound
    dBFS = detection_frames[-1].dBFS    
    previous_dBFS = dBFS if detection_frames[-1].previous_frame is None else detection_frames[-1].previous_frame.dBFS
    if detected and not previously_detected:
        dBFS_delta = abs(dBFS - previous_dBFS)
        detection_state.current_dBFS_threshold = dBFS
        #detection_state.current_zero_crossing_threshold = detection_frames[-1].zero_crossing
    
    # Reset the dBFS to a higher level to fix outset detection
    before_previous_detected = len(detection_frames) > 2 and detection_frames[-3].positive
    if detected and previously_detected and not before_previous_detected and detection_state.current_dBFS_threshold == previous_dBFS:
        dBFS_delta = abs(dBFS - previous_dBFS)
    #    print( "UPDATE DETECTION FROM " + str(detection_state.current_dBFS_threshold) + " TO " + str(dBFS + dBFS_delta / 2 ) )
        detection_state.current_dBFS_threshold = dBFS + dBFS_delta
    
    #print( detection_state.current_dBFS_threshold )
    detected_label = detection_frames[-1].label
    if detected:
        current_occurrence.append(detection_frames[-1])
    else:
        false_occurrence.append(detection_frames[-1])
        detection_state.current_dBFS_threshold = 0
        detection_state.current_zero_crossing_threshold = 0        
    
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
    
    return audioFrames, detection_state, detection_frames, current_occurrence, false_occurrence

def generate_override_dict(override_file):
    lines = []
    with open(override_file, "r") as f:
        lines = f.readlines()
        
    override_dict = {}
    for line in lines:
        if "=" in line:
            items = line.split("=")
            if len(items) == 2:
                value = items[1].strip().lower()
                key = items[0].strip()
                if key.endswith("dbfs"):
                    value = int(value)
                override_dict[key] = value
    return override_dict

detection_count = 0
dBFS_thresholds = []

def determine_detection_frame(index, detection_state, audioFrames, detection_frames) -> DetectionFrame:
    global dBFS_thresholds
    global detection_count
    detected = False
    if( len( audioFrames ) >= SLIDING_WINDOW_AMOUNT ):
        audioFrames = audioFrames[-SLIDING_WINDOW_AMOUNT:]
        
        byteString = b''.join(audioFrames)
        wave_data = np.frombuffer( byteString, dtype=np.int16 )
        power = determine_power( wave_data )
        dBFS = determine_dBFS( wave_data )
        zc = determine_zero_crossing_count( wave_data )

        filtered_dBFS = dBFS#determine_dBFS( high_pass_filter( wave_data ) )
        
        previous_frame = None if len(detection_frames) == 0 else detection_frames[-1]
        zc_delta = 0
        dBFS_delta = 0
        if previous_frame:
            zc_delta = abs(zc - previous_frame.zero_crossing) * (-1 if previous_frame.zero_crossing > zc else 1)
            dBFS_delta = abs(dBFS - previous_frame.dBFS) * (-1 if previous_frame.dBFS > dBFS else 1)
        mfsc_shape = determine_mfsc_shape( wave_data, RATE )
        
        scale = 0.5
        if detection_state.expected_noise_floor != 0:
            max_dBFS = detection_state.expected_noise_floor + detection_state.expected_snr
            scale = abs( dBFS - max_dBFS ) / detection_state.expected_snr
        
        #mfsc_shape *= scale
        distance = determine_euclidean_dist( mfsc_shape, True )
        
        #print( "Index: " + str(index * 15 ) + " DISTANCE " + str(distance) )

        # Attempt to detect a label
        detected_label = BACKGROUND_LABEL
        frame = DetectionFrame(index - 1, detection_state.ms_per_frame, False, power, dBFS, filtered_dBFS, zc, distance, mfsc_shape, BACKGROUND_LABEL)
        
        #( dBFS_delta / detection_state.expected_snr ) * distance        
        #print( "Index: " + str(index * 15 ) + " DISTANCE " + str(onset_value) + (" ONSET!!" if onset_value > 0 else "") )
        likely_threshold = dBFS if len(dBFS_thresholds) == 0 else np.percentile(dBFS_thresholds, 80)
        detected = False
        dBFS_threshold = detection_state.current_dBFS_threshold
        #print( "DBFS THRESHOLD", dBFS_threshold )
        if distance > 4 and (dBFS_threshold == 0 or dBFS_threshold == None):
            dBFS_threshold = dBFS
            detection_state.current_dBFS_threshold = dBFS_threshold
            dBFS_thresholds.append( dBFS )
            #print( "ONSET!" )
            #print( "Index: " + str(index * 15 ) + " DELTA!" + str(distance), " ONSET!!" )
        elif dBFS_threshold < 0:
            dBFS_threshold = dBFS_threshold if dBFS >= dBFS_threshold else 0
            #if dBFS_threshold == 0:
                #print( "OUTSET" )

        
        #print( likely_threshold )
        #onset_threshold = distance#9 * ( 1 / max(1, detection_state.expected_snr ) )

        frame.previous_frame = previous_frame

        for label in detection_state.labels:
            if is_detected(detection_state, frame, label):
            #if detected:
                detected = True
                label.ms_detected += detection_state.ms_per_frame
                frame.positive = detected
                frame.label = label.label
                break

        #print( "Index: " + str(index * 15 ) + " ZCC " + str(zc_delta) + " " + (" - X " if frame.positive else "" )  )

        return frame
    else:
        return DetectionFrame(index - 1, detection_state.ms_per_frame, detected, 0, 0, 0, 0, 0, [], BACKGROUND_LABEL)

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
        
        # Recalculate the MS detection and duration type
        for label in detection_state.labels:
            label.ms_detected = 0
            label.duration_type = determine_duration_type(label, frames)
            if detection_state.override_labels is not None:
                for override_label in detection_state.override_labels:
                    if label.label == override_label.label:
                        label.min_dBFS = label.min_dBFS if override_label.min_dBFS <= -96 else override_label.min_dBFS
                        label.duration_type = label.duration_type if not override_label.duration_type else override_label.duration_type

        for index, frame in enumerate(frames):
            detected = frame.positive
            for label in detection_state.labels:
                if detected:#is_detected(detection_state, frame, label):
            #        detected = True
            #        label.ms_detected += detection_state.ms_per_frame
                    current_label = label
                    break
        
            # Do a secondary pass if the previous label was negative
            # As we can use its thresholds for correcting late starts
            mending_offset = 0
            if detected and not frames[index - 1].positive:
                 for label in detection_state.labels:
                     if current_label.label == label.label and is_detected_secondary(detection_state, frame, label):
                        label.ms_detected += detection_state.ms_per_frame
                        frames[index - 1].label = current_label.label
                        frames[index - 1].positive = True
                        mending_offset = -1
                        if len(false_occurrence) > 0:
                            false_occurrence.pop()
                         
                         # Only do three frames of late start fixing as longer late starts statistically do not seem to occur
                        if not frames[index - 2].positive and is_detected_secondary(detection_state, frames[index - 1], label):
                            label.ms_detected += detection_state.ms_per_frame
                            frames[index - 2].label = current_label.label
                            frames[index - 2].positive = True
                            mending_offset = -2
                            if len(false_occurrence) > 0:
                                false_occurrence.pop()
        
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
    total_average_mel_data = get_average_mel_data([event.average_mel_data for event in events if len(event.average_mel_data) > 0])
    distance_array = []
    
    valid_event_dict = {}
    for event_index, event in enumerate(events):
        if len(event.average_mel_data) > 0:
            distance = np.linalg.norm(np.array(event.average_mel_data) - np.array(total_average_mel_data))
            distance_array.append(distance)
            valid_event_dict[event_index] = distance
    
    # FILTER OUT FINAL DISCREPANCIES
    std_distance = np.std(distance_array)
    average_distance = np.mean(distance_array)
    distance_without_outliers = [dist for dist in distance_array if dist <= average_distance + std_distance]

    std_distance = max(1.5, np.std(distance_without_outliers))
    average_distance = np.mean(distance_without_outliers)
    filtered_events = []
    
    std_ratio = round(( detection_state.expected_snr - 5 ) * 0.2) * 0.5
    for event_index, event in enumerate(events):
        if event_index in valid_event_dict.keys() and valid_event_dict[event_index] < average_distance + std_distance * std_ratio:
            filtered_events.append(event)
       # Change the frames to be silence instead
        else:
            for event_frame in event.frames:
                frames[event_frame.index].label = BACKGROUND_LABEL
                frames[event_frame.index].positive = False

    print( str( len(events) - len(filtered_events) ) + " EVENTS FILTERED!")
    persist_srt_file( output_filename, filtered_events )
    
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
    dBFS_frames = [x.dBFS for x in detection_frames if x.dBFS > -70 and x.dBFS != 0]
    filtered_dBFS_frames = dBFS_frames#[x.filtered_dBFS for x in detection_frames if x.dBFS > -70 and x.filtered_dBFS != 0]
    if len(dBFS_frames) == 0:
        dBFS_frames = [0]
    if len(filtered_dBFS_frames) == 0:
        filtered_dBFS_frames = [0]

    std_dBFS = np.std(dBFS_frames)

    minimum_filtered_dBFS = np.min(filtered_dBFS_frames)
    #std_filtered_dBFS = np.std(filtered_dBFS_frames)
    
    minimum_dBFS = np.percentile(dBFS_frames, 3)
    max_dBFS = np.percentile(dBFS_frames, 97)
    
    #print( "DISTANCE!", abs(max_dBFS - minimum_dBFS ) )
    #max_filtered_dBFS = np.max(filtered_dBFS_frames)    
    
    # Calculate the dBFS threshold for the signal using some averaging techniques
    # I found this specific threshold calculation based on some files without and with added large amount of noise
    # To emulate poorer microphones
    #dBFS_threshold = (abs(max_dBFS) - abs(minimum_filtered_dBFS)) / 2 - ( std_filtered_dBFS - std_dBFS ) / 2 # Strategy 1
        
    # To account for signals that have a low average and high std value
    # We calculate a ratio in which the std will be removed using Strategy 8 to ensure
    # We do not put our threshold for these signals too high
    #std_ratio = abs(round((abs(np.mean(dBFS_frames)) - std_dBFS * 2 ) / 10 ) )
    #dBFS_threshold -= (abs(np.mean(filtered_dBFS_frames)) - std_dBFS * std_ratio ) # Strategy 8
    #dBFS_threshold = dBFS_threshold / 2 - ( max_filtered_dBFS - max_dBFS ) * 2 # Strategy 14
    
    difference = abs(minimum_dBFS - minimum_filtered_dBFS)
    detection_state.expected_snr = abs(minimum_dBFS - max_dBFS )#std_dBFS * 2#math.floor(((std_dBFS + std_filtered_dBFS) / 2) * 2)
    detection_state.expected_noise_floor = minimum_dBFS#minimum_dBFS + std_dBFS / 2
    
    dBFS_threshold = detection_state.expected_noise_floor + std_dBFS / 2
    
    # Determine the secondary threshold based on the difference in the signal noise
    secondary_threshold = std_dBFS

    for label in detection_state.labels:
        # Recalculate the duration type every 15 seconds for the first minute
        if len(detection_frames) % round(15 / RECORD_SECONDS) == 0 and len(detection_frames) <= 60 / RECORD_SECONDS:
            label.duration_type = determine_duration_type(label, detection_frames)

        label.min_dBFS = dBFS_threshold # -28 is expected # TODO CALCULATE THRESHOLD PER LABEL
        label.min_secondary_dBFS = label.min_dBFS - secondary_threshold
    detection_state.latest_dBFS = detection_frames[-1].dBFS
    previous_dBFS = detection_state.latest_dBFS if len(detection_frames) == 1 else detection_frames[-2].dBFS
    detection_state.latest_delta = abs(detection_frames[-1].dBFS - previous_dBFS) * ( 1 if previous_dBFS < detection_frames[-1].dBFS else -1 )

    # Override the detection by manual overrides
    if detection_state.override_labels is not None:
        for label in detection_state.labels:
            for override_label in detection_state.override_labels:
                if label.label == override_label.label:
                    label.min_dBFS = label.min_dBFS if override_label.min_dBFS <= -96 else override_label.min_dBFS
                    label.min_secondary_dBFS = label.min_dBFS - secondary_threshold
                    label.duration_type = label.duration_type if not override_label.duration_type else override_label.duration_type
    
    return detection_state

# Approximately determine whether the label in the stream is discrete or continuous
# Discrete sounds are from a single source event like a click, tap or a snap
# Whereas continuous sounds have a steady stream of energy from a source
def determine_duration_type(label: DetectionLabel, detection_frames: List[DetectionFrame]) -> str:
    label_events = [x for x in detection_frames_to_events(detection_frames) if x.label == label.label]
    if len(label_events) < 4:
        return ""
    else:
        # #1 - The assumption here is that discrete sounds cannot vary in length much as you cannot elongate the sound of a click for example
        # So if the length doesn't vary much, we assume discrete over continuous
        lengths = [x.end_ms - x.start_ms for x in label_events]
        
        # Assumption #2 - The envelope of discrete sounds vs continous sounds is very distinct
        # Discrete sounds spike up rapidly and move down quickly as well, whereas continuous noises are more gradual
        # We use an STD of the mean of the event dBFS' to determine whether or not we should determine discrete or continuous
        # In an experiment with 6 noises ( 3 discrete, 3 continous ) the threshold of 1.5 was found to be a good distinction        

        std_of_average_dBFS = np.std([x.average_dBFS for x in label_events])
        return "discrete" if std_of_average_dBFS > 2 else "continuous"

def detection_frames_to_events(detection_frames: List[DetectionFrame]) -> List[DetectionEvent]:
    events = []
    current_label = BACKGROUND_LABEL
    current_frames = []
    for frame in detection_frames:
        label_changing = current_label != frame.label
        current_label = frame.label
        if frame.label != BACKGROUND_LABEL:
            current_frames.append( frame )
        
        if label_changing and frame.label == BACKGROUND_LABEL:
            if len(current_frames) > 0:
                event = same_frames_to_detection_frames(current_frames)
                events.append( event )
                current_frames = []
            
    if len(current_frames) > 0:
        event = same_frames_to_detection_frames(current_frames)
        events.append( event )
        current_frames = []

    return events
    
def same_frames_to_detection_frames(current_frames: List[DetectionFrame]) -> DetectionEvent:    
    average_mel_data = get_average_mel_data([frame.mel_data for frame in current_frames])
    average_dBFS = np.mean([frame.dBFS for frame in current_frames])
    return DetectionEvent(current_frames[-1].label, current_frames[0].index, current_frames[-1].index, \
        (current_frames[0].index) * current_frames[0].duration_ms, (current_frames[-1].index + 1) * current_frames[-1].duration_ms, average_dBFS, average_mel_data, current_frames)

# Calculate the average event sound so we can use it to pick out outliers
def get_average_mel_data(mel_data_frames: List[List[float]]) -> List[List[float]]:
    total_mel_data = []
    for mel_data in mel_data_frames:
        if len(mel_data) > 0:
            if len(total_mel_data) == 0:
                total_mel_data = mel_data
            else:
                for index, mel_window in enumerate(mel_data):
                    totaled_mel_window = []
                    for item_index, item in enumerate(mel_window):
                        if item_index < 4:
                            totaled_mel_window.append(0)
                        else:
                            totaled_mel_window.append(total_mel_data[index][item_index] + item)
                    total_mel_data[index] = totaled_mel_window
    data = np.multiply(1 / max(1, len(mel_data_frames)), total_mel_data)
    for window_index, window in enumerate(data):
        for item_index, item in enumerate(window):
            data[window_index][item_index] = item if item > 0 else 0
    
    return data

def auto_decibel_detection(power, dBFS, distance, dBFS_threshold):
    return dBFS >= dBFS_threshold
    
def auto_secondary_decibel_detection(power, dBFS, distance, dBFS_threshold):
    return dBFS >= dBFS_threshold

detected_dBFS = 0

def is_detected(detection_state, frame, label):
    strategy = detection_state.strategy
    power = frame.power
    dBFS = frame.dBFS
    filtered_dBFS = frame.filtered_dBFS
    previous_dBFS = dBFS if frame.previous_frame is None else frame.previous_frame.dBFS
    dBFS_delta = abs(dBFS - previous_dBFS) * ( 1 if previous_dBFS < dBFS else -1 )

    zero_crossing_count = frame.zero_crossing
    previous_zcc = zero_crossing_count if frame.previous_frame is None else frame.previous_frame.zero_crossing
    zc_delta = abs(zero_crossing_count - previous_zcc) * ( 1 if previous_zcc < zero_crossing_count else -1 )

    distance = frame.euclid_dist
    estimated_threshold = label.min_dBFS
    expected_snr = detection_state.expected_snr
    
    threshold = detection_state.current_dBFS_threshold if detection_state.current_dBFS_threshold < 0 else 0
    
    global detected_dBFS
    #if dBFS_delta > 10.6:
    #    detected_dBFS = dBFS - ( dBFS_delta / 2 )
        #print( "    YES RATE CHANGE +" + str(dBFS_delta) + " to " + str(dBFS) )
    #    return True
    #if dBFS_delta < -5.3:
    #    detected_dBFS = 0
        #print( "    NO! RATE CHANGE +" + str(dBFS_delta) + " to " + str(dBFS) )
    #    return False
    if "auto_dBFS" in strategy:
        detected = auto_decibel_detection(power, dBFS, distance, threshold)
        #if not detected and dBFS > detected_dBFS:
        #   detected = True
           #print( "WAS NO BUT SHOULD BE YES! " )
        #if not detected and dBFS < detected_dBFS:
        #   detected_dBFS = 0
            #print( "NO! " + str(dBFS_delta) + " " + str(dBFS) )
        #else:
        #    if detected_dBFS == 0:
        #        detected_dBFS = dBFS
            #print( "    YES " + str(dBFS) + " over " + str(detected_dBFS) )
            
        return detected
    elif "auto_avg_dBFS" in strategy:
        return auto_decibel_detection(power, (dBFS + filtered_dBFS) / 2, distance, estimated_threshold)
    elif "auto_weighted_dBFS" in strategy:
        signal_weight = expected_snr / snr_cutoff if expected_snr < snr_cutoff else 1
        filtered_weight = 1 - signal_weight
        return auto_decibel_detection(power, (dBFS * signal_weight + filtered_dBFS * filtered_weight), distance, estimated_threshold)


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

def is_detected_secondary(detection_state, frame, label):
    global detected_dBFS
    strategy = detection_state.strategy
    power = frame.power
    dBFS = frame.dBFS
    previous_dBFS = dBFS if frame.previous_frame is None else frame.previous_frame.dBFS
    dBFS_delta = abs(dBFS - previous_dBFS) * ( 1 if previous_dBFS < dBFS else -1 )

    zero_crossing_count = frame.zero_crossing
    previous_zcc = zero_crossing_count if frame.previous_frame is None else frame.previous_frame.zero_crossing
    zc_delta = abs(zero_crossing_count - previous_zcc) * ( 1 if previous_zcc < zero_crossing_count else -1 )

    filtered_dBFS = frame.filtered_dBFS
    distance = frame.euclid_dist
    estimated_threshold = label.min_dBFS
    expected_snr = detection_state.expected_snr
    secondary_threshold = label.min_secondary_dBFS
    
    if "secondary" not in strategy:
        return False
    elif "secondary_dBFS" in strategy:
        return auto_secondary_decibel_detection(power, dBFS, distance, estimated_threshold - 7)
    elif "secondary_avg_dBFS" in strategy:
        return auto_secondary_decibel_detection(power, ( dBFS + filtered_dBFS ) / 2, distance, estimated_threshold)
    elif "secondary_weighted_dBFS" in strategy:
       signal_weight = expected_snr / snr_cutoff if expected_snr < snr_cutoff else 1
       filtered_weight = 1 - signal_weight
       return auto_secondary_decibel_detection(power, ( dBFS * signal_weight + filtered_dBFS * filtered_weight ), distance, estimated_threshold)
    elif "secondary_std_dBFS" in strategy:
        return auto_secondary_decibel_detection(power, dBFS , distance, secondary_threshold)

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
