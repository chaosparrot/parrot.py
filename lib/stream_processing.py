from .typing import DetectionLabel, DetectionFrame, DetectionEvent, DetectionState
from config.config import BACKGROUND_LABEL, RECORD_SECONDS, SLIDING_WINDOW_AMOUNT, RATE, CURRENT_VERSION, CURRENT_DETECTION_STRATEGY
from typing import List
import wave
import math
import numpy as np
from .signal_processing import determine_power, determine_dBFS, determine_log_mels, determine_euclidean_dist
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
    detection_state = DetectionState(detection_strategy, "recording", ms_per_frame, 0, True, 0, 0, 0, 0, detection_labels, None, [])
    
    # Add manual overrides if the override file exists
    if override_file is not None and os.path.exists(override_file):
        override_dict = generate_override_dict(override_file)
    
        for override_label in labels:
            duration_type = ""
            duration_type_key = (override_label + "_duration_type").lower()
            if duration_type_key in override_dict and override_dict[duration_type_key].lower() in ["discrete", "continuous"]:
                duration_type = override_dict[duration_type_key].lower()
                
            min_dBFS = -96
            min_dBFS_key = (override_label + "_min_dbfs").lower()
            if min_dBFS_key in override_dict and override_dict[min_dBFS_key] < 0:
                min_dBFS = override_dict[min_dBFS_key]
            
            override_labels.append(DetectionLabel(override_label, 0, 0, duration_type, 0, min_dBFS, 0, 0, 0))    
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
    current_detection_frame = determine_detection_frame(index, detection_state, audioFrames, detection_frames)    
    detection_frames.append(current_detection_frame)
    detected = detection_frames[-1].positive
    previously_detected = len(detection_frames) > 1 and detection_frames[-2].positive

    onset_detected = current_detection_frame.onset
    previous_onset_detected = False if len(detection_frames) == 1 else detection_frames[-2].onset

    # Determine threshold of the current sound based on onset detection
    if previous_onset_detected and not onset_detected and not previously_detected:
        detection_state.current_dBFS_threshold = detection_frames[index - 2].dBFS
        print( index * 15, "SETTING CURRENT THRESHOLD TO", detection_state.current_dBFS_threshold )

        # Ensure that we make use of an upper bound - So that we do not have issues with very soft spikes being detected
        if detection_state.upper_bound_dBFS_threshold != 0 and detection_state.current_dBFS_threshold < detection_state.upper_bound_dBFS_threshold:
            detection_state.current_dBFS_threshold = detection_state.upper_bound_dBFS_threshold
            print( "RESETTING CURRENT THRESHOLD TO UPPER BOUND!", detection_state.upper_bound_dBFS_threshold )

        # Attempt to detect again using the new threshold
        if not detected:
            for label in detection_state.labels:
                if is_detected(detection_state, current_detection_frame, label):
                    detected = True
                    detection_frames[-1].positive = detected
                    detection_frames[-1].label = label.label
                    label.ms_detected += detection_state.ms_per_frame
                    break
    
    detected_dBFS_values = []

    # Once we have achieved a peak, the sound must be undetected after it has fallen below 30% of its peak
    if detected or previously_detected:
        starting_range = len(detection_frames) - 1 if detected else len(detection_frames) - 2
        for detected_index in range(starting_range, 0, -1):
            if detection_frames[detected_index].positive:
                detected_dBFS_values.append(detection_frames[detected_index].dBFS)
            else:
                break

        used_dBFS_threshold = detection_state.current_dBFS_threshold
        dynamic_range_sound = abs(np.percentile(detected_dBFS_values, 90) - used_dBFS_threshold)
        if detected and abs(current_detection_frame.dBFS - used_dBFS_threshold) < dynamic_range_sound * 0.2 \
            and current_detection_frame.spectral_flux < detection_state.spectral_onset_threshold:
            detected = False

        if detected and not previously_detected:
            print( index * 15, "DETECTED!" )
        elif not detected and previously_detected:
            print( index * 15, "EXIT!" )

        # Short burst detection at the start
        # If we exit too early just make sure that descent wasn't bigger than our error margin
        if len(detected_dBFS_values) == 1 and not detected:
            detection_state.current_dBFS_threshold -= detection_state.dBFS_error_margin
            print( "ERROR MARGIN DOWNWARD!" )
            detected = False
            for label in detection_state.labels:
                if is_detected(detection_state, current_detection_frame, label):
                    print( "SHORT BURST PROTECTION!" )
                    detected = True
                    label.ms_detected += detection_state.ms_per_frame
                    break
        
        # Properly clean up state from detected to not-detected
        # Remove the detected milliseconds from the recorded values
        if not detected and current_detection_frame.positive:
            for label in detection_state.labels:
                if label.label == current_detection_frame.label:
                    label.ms_detected -= detection_state.ms_per_frame
                    break
            detection_frames[-1].positive = detected
            detection_frames[-1].label = BACKGROUND_LABEL
        
        # If we have repairs, ensure they are properly set
        elif detected and not current_detection_frame.positive:
            detection_frames[-1].positive = detected

        # Add a known dBFS exit valley from the current detection streak
        if not detected and previously_detected:
            new_dBFS_valley = np.percentile(detected_dBFS_values, 20)
            detection_state.dBFS_valleys.append(new_dBFS_valley)

            # Reset the current dBFS threshold to mark the end of a sound
            detection_state.current_dBFS_threshold = 0
        
    detected_label = detection_frames[-1].label
    if detected:
        current_occurrence.append(detection_frames[-1])
    else:
        false_occurrence.append(detection_frames[-1])
    
    # Recalculate the noise floor / signal strength every 15 frames
    # For performance reason and because the statistical likelyhood of things changing every 150ms is pretty low
    if len(detection_frames) % 15 == 0:
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
                print( "MENDED BECAUSE TOO SHORT!", total_mended_frames)
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

def standardize(frames: np.array) -> np.array:
    mean = np.mean(frames)
    std = np.std(frames)
    if std > 0:
        return (frames - mean) / std
    else:
        return frames - mean

detection_count = 0
dBFS_thresholds = []

def detect_onset(index, dBFS, spectral_flux, detection_state, detection_frames) -> bool:
    # Onset detection by finding the peak in the last N frames
    onset_detected = False
    if detection_state.spectral_onset_threshold is not None and spectral_flux >= detection_state.spectral_onset_threshold:
        # On-line
        spectral_flux_to_check_peak = [spectral_flux]

        # Find a peak within the last three frames
        if index > len(detection_frames) and index > 2:
            spectral_flux_to_check_peak.extend([frame.spectral_flux for frame in detection_frames[index - 3:]])            

        # Post-processing - Find a peak within the last 7 frames ( three frames back and three frames forward )
        elif index > 3 and index < len(detection_frames) - 3:
            spectral_flux_to_check_peak.extend([frame.spectral_flux for frame in detection_frames[index - 3: index + 3]])

        onset_detected = spectral_flux == max(spectral_flux_to_check_peak)
        if onset_detected and index > 2 and dBFS < detection_frames[index - 2].dBFS:
            onset_detected = False
    return onset_detected

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

        log_mels = determine_log_mels( wave_data, RATE )
        spectral_flux = determine_euclidean_dist( log_mels, True )
        onset_detected = detect_onset( index, dBFS, spectral_flux, detection_state, detection_frames )
        
        # Attempt to detect a label
        detected_label = BACKGROUND_LABEL
        frame = DetectionFrame(
            index - 1,
            detection_state.ms_per_frame, 
            False,
            onset_detected,
            power,
            dBFS,
            log_mels,
            spectral_flux,
            detected_label
        )
        
        for label in detection_state.labels:
            if is_detected(detection_state, frame, label):
                detected = True
                label.ms_detected += detection_state.ms_per_frame
                frame.positive = detected
                frame.label = label.label
                break
        
        return frame
    else:
        return DetectionFrame(
            index - 1,
            detection_state.ms_per_frame,
            detected,
            False,
            0,
            0,
            [],
            0,
            BACKGROUND_LABEL
        )

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

        # Set the current sound threshold
        detection_state.current_dBFS_threshold = detection_state.upper_bound_dBFS_threshold
        for index, frame in enumerate(frames):
            detected = frame.positive
            for label in detection_state.labels:
                if is_detected(detection_state, frame, label):
                    detected = True
                    label.ms_detected += detection_state.ms_per_frame
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
    total_average_mel_data = get_average_log_mels([event.average_log_mels for event in events if len(event.average_log_mels) > 0])
    distance_array = []
    
    valid_event_dict = {}
    for event_index, event in enumerate(events):
        if len(event.average_log_mels) > 0:
            distance = np.linalg.norm(np.array(event.average_log_mels) - np.array(total_average_mel_data))
            distance_array.append(distance)
            valid_event_dict[event_index] = distance
    
    # FILTER OUT FINAL DISCREPANCIES
    std_distance = np.std(distance_array)
    average_distance = np.mean(distance_array)
    distance_without_outliers = [dist for dist in distance_array if dist <= average_distance + std_distance]

    std_distance = max(1.5, np.std(distance_without_outliers))
    average_distance = np.median(distance_without_outliers)
    filtered_events = []
    
    for event_index, event in enumerate(events):
        if event_index in valid_event_dict.keys() and valid_event_dict[event_index] < average_distance + std_distance:
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
    spectral_flux_max = np.percentile([frame.spectral_flux for frame in frames], 95)
    spectral_flux_min = np.percentile([frame.spectral_flux for frame in frames], 5)
    dBFS_max = np.percentile([frame.dBFS for frame in frames], 95)
    dBFS_min = np.percentile([frame.dBFS for frame in frames], 5)
    spectral_onset_threshold = (spectral_flux_max - spectral_flux_min) * 0.5
    detected_ms = 0
    if output_wave_file is not None:
        frames_to_write = round( RATE * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT )
        sample_width = 2# 16 bit = 2 bytes
        detection_audio_frames = []
        previous_onset_detected = False
        previous_detected = False
        previous_dB_threshold = 0
        average_dB_threshold = 0
        std_dB_threshold = 0

        dBFS_safety_margin = abs(dBFS_min - dBFS_max) / 25

        known_valleys = []
        current_detection = []
        for index, frame in enumerate(frames):
            highest_amp = 65536 / 10

            # Detect the peak of N frames if it is above 25% of the maximum value
            onset_detected = False
            spectral_flux = frame.spectral_flux
            if spectral_flux >= spectral_onset_threshold:
                if index > 2 and index < len(frames) - 3:
                    onset_detected = spectral_flux == max([frame.spectral_flux for frame in frames[index - 3: index + 3]])
                    #print( "ONSET :D", onset_detected, index, [frame.spectral_flux for frame in frames[index - 3: index + 3]], max([frame.spectral_flux for frame in frames[index - 3: index + 3]]) )
                else:
                    onset_detected = True
            if onset_detected and index > 1 and frame.dBFS < frames[index - 1].dBFS:
                onset_detected = False
            #    print( "NO ONSET >:(")

            # TODO!!!!
            if previous_onset_detected and not onset_detected and not previous_detected:
                previous_dB_threshold = frames[index - 1].dBFS
            #    if previous_dB_threshold < -35.1425543499734:
            #        previous_dB_threshold = -35.1425543499734
            #    
            #    # TODO POST PROCESSING WITH FOUDN PREVIOUS DB THRESHOLD BASED ON KNOWN VALLEYS!!!
                if average_dB_threshold != 0 and previous_dB_threshold < average_dB_threshold - std_dB_threshold:
                    previous_dB_threshold = average_dB_threshold - std_dB_threshold

            previous_onset_detected = onset_detected
            detected = frame.dBFS >= previous_dB_threshold

            # Short burst detection at the start
            if len(current_detection) == 1 and not detected:
                previous_dB_threshold -= dBFS_safety_margin
                detected = frame.dBFS >= previous_dB_threshold
                #print( "SAVED BECAUSE OF SAFETY MARGIN", dBFS_safety_margin, detected)

            # Once we have achieved a peak, the sound must be undetected after it has fallen below 10% of its peak
            if detected and previous_detected:
                dynamic_range_sound = abs(np.percentile(current_detection, 90) - previous_dB_threshold)
                if abs(frame.dBFS - previous_dB_threshold) < dynamic_range_sound * 0.1 and spectral_flux < spectral_onset_threshold:
                    detected = False
            #        #print( "BECAUSE " + ("RANGE" if abs(frame.dBFS - previous_dB_threshold) < dynamic_range_sound * 0.2 else "SPECTRAL FLUX"))

            if not detected:
                if len(current_detection) > 0:
                    known_valleys.append(np.percentile(current_detection, 10))
                    #print( str( 15 + len(current_detection) * 15) + "ms recorded" )
                    current_detection = []
                average_dB_threshold = 0 if len(known_valleys) < 10 else np.median(known_valleys)#( np.max(known_valleys) - np.min(known_valleys) ) / 2
                std_dB_threshold = 0 if len(known_valleys) < 10 else np.std(known_valleys)
                previous_dB_threshold = 0 if len(known_valleys) < 10 else average_dB_threshold - std_dB_threshold / 2
            #    #if previous_detected:
            #        #print( "ESCAPE!", max([frame.spectral_flux for frame in frames[index - 3: index + 3]]) if index > 2 else 0, frame.spectral_flux, "CURRENT DBFS", frame.dBFS, "Threshold", previous_dB_threshold, "Avg", average_dB_threshold - std_dB_threshold / 2)                    
            else:
                #print( "DBFS CHANGE", index * 15, abs(frame.dBFS - frames[index - 1].dBFS) * (1 if frame.dBFS > frames[index - 1].dBFS else -1), abs(frame.dBFS - previous_dB_threshold), previous_dB_threshold )
                #print( "FLUX!", max([frame.spectral_flux for frame in frames[index - 3: index + 3]]) if index > 2 else 0, frame.spectral_flux, "CURRENT DBFS", frame.dBFS, "Threshold", previous_dB_threshold, "Avg", average_dB_threshold, std_dB_threshold )
                current_detection.append(frame.dBFS)
            
            if detected:
                detected_ms += 15

            #if index > 1:
            #    spectral_flux += frames[index - 2].spectral_flux
            #print( index * 15, spectral_flux )
            #print( frame.index * 15, frame.positive )

            signal_strength = highest_amp if frame.positive else 0# * (frame.spectral_flux / spectral_flux_max)

            detection_signal = np.full(int(frames_to_write / sample_width), int(signal_strength))
            detection_signal[::2] = 0
            detection_signal[::3] = 0
            detection_signal[::5] = 0
            detection_signal[::7] = 0
            detection_signal[::9] = 0
            detection_audio_frames.append( detection_signal )
            #previous_detected = detected
        output_wave_file.writeframes(b''.join(detection_audio_frames))
        output_wave_file.close()

    print( detected_ms )
    detection_state.state = "recording"
    return frames

def determine_detection_state(detection_frames: List[DetectionFrame], detection_state: DetectionState) -> DetectionState:
    dBFS_frames = [x.dBFS for x in detection_frames]
    
    # Calculate the onset thresholds using spectral flux
    spectral_flux_max = np.percentile([frame.spectral_flux for frame in detection_frames], 95)
    spectral_flux_min = np.percentile([frame.spectral_flux for frame in detection_frames], 5)
    detection_state.spectral_onset_threshold = (spectral_flux_max - spectral_flux_min) * 0.5

    # Calculate the signal variance
    std_dBFS = np.std(dBFS_frames)
    detection_state.expected_snr = std_dBFS * 2
    detection_state.expected_noise_floor = np.percentile(dBFS_frames, 10)
    
    # Determine an error margin of about 4% of the rough dBFS range
    dBFS_max = np.percentile([frame.dBFS for frame in detection_frames], 95)
    dBFS_min = np.percentile([frame.dBFS for frame in detection_frames], 5)
    detection_state.dBFS_error_margin = abs(dBFS_min - dBFS_max) / 25

    # Determine a lower bound of dBFS threshold based on the known valleys determined by the onset detection
    if len(detection_state.dBFS_valleys) >= 10:
        median_dB_threshold = np.median(detection_state.dBFS_valleys)
        std_dB_threshold = np.std(detection_state.dBFS_valleys)        
        detection_state.upper_bound_dBFS_threshold = median_dB_threshold - std_dB_threshold
    else:
        detection_state.upper_bound_dBFS_threshold = 0

    for label in detection_state.labels:
        # Recalculate the duration type every 15 seconds for the first minute
        if len(detection_frames) % round(15 / RECORD_SECONDS) == 0 and len(detection_frames) <= 60 / RECORD_SECONDS:
            label.duration_type = determine_duration_type(label, detection_frames)

        label.min_dBFS = detection_state.upper_bound_dBFS_threshold
        label.min_secondary_dBFS = label.min_dBFS
    detection_state.latest_dBFS = detection_frames[-1].dBFS
    previous_dBFS = detection_state.latest_dBFS if len(detection_frames) == 1 else detection_frames[-2].dBFS
    detection_state.latest_delta = abs(detection_frames[-1].dBFS - previous_dBFS) * ( 1 if previous_dBFS < detection_frames[-1].dBFS else -1 )

    # Override the detection by manual overrides
    if detection_state.override_labels is not None:
        for label in detection_state.labels:
            for override_label in detection_state.override_labels:
                if label.label == override_label.label:
                    label.min_dBFS = label.min_dBFS if override_label.min_dBFS <= -96 else override_label.min_dBFS
                    label.min_secondary_dBFS = label.min_dBFS
                    label.duration_type = label.duration_type if not override_label.duration_type else override_label.duration_type
                    label.overridden = True
    
    return detection_state

# Approximately determine whether the label in the stream is discrete or continuous
# Discrete sounds are from a single source event like a click, tap or a snap
# Whereas continuous sounds have a steady stream of energy from a source
def determine_duration_type(label: DetectionLabel, detection_frames: List[DetectionFrame]) -> str:
    label_events = [x for x in detection_frames_to_events(detection_frames) if x.label == label.label]
    if len(label_events) < 4:
        return ""
    else:        
        # Assumption - The envelope of discrete sounds vs continous sounds is very distinct
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
    average_mel_data = get_average_log_mels([frame.log_mels for frame in current_frames])
    average_dBFS = np.mean([frame.dBFS for frame in current_frames])
    return DetectionEvent(current_frames[-1].label, current_frames[0].index, current_frames[-1].index, \
        (current_frames[0].index) * current_frames[0].duration_ms, (current_frames[-1].index + 1) * current_frames[-1].duration_ms, average_dBFS, average_mel_data, current_frames)

# Calculate the average event sound so we can use it to pick out outliers
def get_average_log_mels(log_mels: List[List[float]]) -> List[List[float]]:
    total_mel_data = []
    for mel_data in log_mels:
        if len(mel_data) > 0:
            mel_data = standardize(mel_data)
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
    data = np.multiply(1 / max(1, len(log_mels)), total_mel_data)
    for window_index, window in enumerate(data):
        for item_index, item in enumerate(window):
            data[window_index][item_index] = item if item > 0 else 0
    
    return data

def auto_decibel_detection(power, dBFS, distance, dBFS_threshold):
    return dBFS >= dBFS_threshold
    
def auto_secondary_decibel_detection(power, dBFS, distance, dBFS_threshold):
    return dBFS >= dBFS_threshold

detected_dBFS = 0

def is_detected(detection_state: DetectionState, frame: DetectionFrame, label):
    strategy = detection_state.strategy
    power = frame.power
    dBFS = frame.dBFS

    distance = frame.spectral_flux
    if label.overridden:
        threshold = label.min_dBFS
    elif detection_state.current_dBFS_threshold is not None:
        threshold = detection_state.current_dBFS_threshold
    else:
        threshold = label.min_dBFS

    global detected_dBFS
    if "auto_dBFS" in strategy:
        detected = auto_decibel_detection(power, dBFS, distance, threshold)
        return detected

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

def is_detected_secondary(detection_state: DetectionState, frame: DetectionFrame, label):
    global detected_dBFS
    strategy = detection_state.strategy
    power = frame.power
    dBFS = frame.dBFS

    distance = frame.spectral_flux
    if label.overridden:
        threshold = label.min_secondary_dBFS
    elif detection_state.current_dBFS_threshold is not None:
        threshold = detection_state.current_dBFS_threshold
        if "secondary_margin_dBFS" in strategy:
            threshold -= detection_state.dBFS_error_margin
    else:
        threshold = label.min_secondary_dBFS
        if "secondary_margin_dBFS" in strategy:
            threshold -= detection_state.dBFS_error_margin

    # Remove the error margin to make small dips not cause invalid issues

    if "secondary" not in strategy:
        return False
    elif "secondary_dBFS" in strategy:
        return auto_secondary_decibel_detection(power, dBFS, distance, threshold)

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
            if not auto_secondary_decibel_detection(frame.power, frame.dBFS, frame.spectral_flux, label_dBFS_threshold):
                if not "mend_dBFS_30ms" in strategy:
                    return False
                else:
                    total_missed_length_ms += detection_state.ms_per_frame
        if not "mend_dBFS_30ms" in strategy:
            return True
        else:
            return total_missed_length_ms < 30
