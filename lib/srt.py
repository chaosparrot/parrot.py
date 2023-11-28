import time
from config.config import BACKGROUND_LABEL, CURRENT_VERSION
from .typing import TransitionEvent, DetectionEvent, DetectionFrame
from typing import List
import math
import os
import numpy as np

current_v_ending = ".v" + str(CURRENT_VERSION) + ".srt"
manual_ending = ".MANUAL.srt"

def ms_to_srt_timestring( ms: int, include_hours=True):
    if ms <= 0:
        return "00:00:00,000" if include_hours else "00:00,000"    

    if include_hours:
        hours = math.floor(ms / (60 * 60 * 1000))
        ms -= hours * 60 * 60 * 1000
    minutes = math.floor(ms / (60 * 1000))
    ms -= minutes * 60 * 1000
    seconds = math.floor(ms / 1000)
    ms -= seconds * 1000
    return ( "{:02d}".format(hours) + ":" if include_hours else "" ) + "{:02d}".format(minutes) + ":" + "{:02d}".format(seconds) + "," + "{:03d}".format(ms)

def srt_timestring_to_ms( srt_timestring: str):
      ms = int(srt_timestring.split(",")[1])
      ms += int(srt_timestring.split(":")[2].split(",")[0]) * 1000
      ms += int(srt_timestring.split(":")[1]) * 60 * 1000
      ms += int(srt_timestring.split(":")[0]) * 60 * 60 * 1000
      return ms

def persist_srt_file(srt_filename: str, events: List[DetectionEvent]):
    if not srt_filename.endswith(current_v_ending) and not srt_filename.endswith(manual_ending):
        srt_filename += current_v_ending

    # Sort events chronologically first
    events.sort(key = lambda event: event.start_index)    
    with open(srt_filename, 'w') as srt_file:
        for index, event in enumerate(events):
            srt_file.write( str(index + 1) + '\n' )
            srt_file.write( ms_to_srt_timestring(event.start_ms) + " --> " + ms_to_srt_timestring(event.end_ms) + '\n' )
            srt_file.write( event.label + '\n\n' )

def parse_srt_file(srt_filename: str, rounding_ms: int, show_errors: bool = True) -> List[TransitionEvent]:
    transition_events = []
    positive_event_list = []

    if not srt_filename.endswith(".srt"):
        srt_filename += ".srt"

    with open(srt_filename, "r") as srt:
       time_start = 0
       time_end = 0
       type_sound = ""
       for line_index, line in enumerate(srt):
           if not line.strip():
              time_start = 0
              time_end = 0
              type_sound = ""
           elif "-->" in line:
              # Extract time start and end rounded to the window size
              # To give the detection a fair estimate of correctness
              time_pair = [timestring.strip() for timestring in line.split("-->")]
              time_start = math.ceil(srt_timestring_to_ms( time_pair[0] ) / rounding_ms) * rounding_ms
              
              time_end = math.ceil(srt_timestring_to_ms( time_pair[1] ) / rounding_ms) * rounding_ms
           elif not line.strip().isnumeric():
              if type_sound == "":
                  type_sound = line.strip()
                  if time_start < time_end:
                      positive_event_list.append(str(time_start) + "---" + type_sound + "---start")
                      positive_event_list.append(str(time_end) + "---" + type_sound + "---end")
                  elif show_errors:
                      print( ".SRT error at line " + str(line_index) + " - Start time not before end time! Not adding this event - Numbers won't be valid!" )

    # Sort chronologically by time
    positive_event_list.sort(key = lambda event: int(event.split("---")[0]))
    for time_index, time_event in enumerate(positive_event_list):
        # Remove duplicates if found
        if time_index != 0 and len(transition_events) > 0 and transition_events[-1].start_index == math.floor(int(time_event.split("---")[0]) / rounding_ms):  
            if show_errors:
                print( "Found duplicate entry at second " + str(math.floor(int(time_event.split("---")[0]) / rounding_ms) / 1000) + " - Not adding duplicate")
            continue;

        if time_event.endswith("---start"):
            if time_index == 0 and int(time_event.split("---")[0]) > 0:
                transition_events.append( TransitionEvent(BACKGROUND_LABEL, 0, 0) )
            
            ms_start = math.floor(int(time_event.split("---")[0]))
            
            # If the time between the end and start of a new event is 0, then the previous event should be removed
            if len(transition_events) > 0 and ms_start - transition_events[-1].start_ms <= rounding_ms:
                transition_events.pop()
            
            transition_events.append( TransitionEvent(time_event.split("---")[1], math.floor(ms_start / rounding_ms), ms_start) )
        elif time_event.endswith("---end"):
            ms_start = math.floor(int(time_event.split("---")[0]))
            transition_events.append( TransitionEvent(BACKGROUND_LABEL, math.floor(ms_start / rounding_ms), ms_start) )
    
    return transition_events

def count_total_frames(label: str, base_folder: str, rounding_ms: int) -> int:
    frames = 0
    segments_dir = os.path.join(base_folder, "segments")
    if os.path.isdir(segments_dir):
        srt_files = [x for x in os.listdir(segments_dir) if os.path.isfile(os.path.join(segments_dir, x)) and (x.endswith(current_v_ending) or x.endswith(manual_ending))]
        for srt_file in srt_files:
        
            # Skip the file if a manual override file of it exists
            if srt_file.endswith(current_v_ending) and srt_file.replace(current_v_ending, manual_ending) in srt_files:
                continue
            else:
                frames += count_frames_in_srt(label, os.path.join(segments_dir, srt_file), rounding_ms)
    return frames
    
def count_total_silence_frames(base_folder: str, rounding_ms: int) -> int:
    frames = 0
    segments_dir = os.path.join(base_folder, "segments")
    if os.path.isdir(segments_dir):
        srt_files = [x for x in os.listdir(segments_dir) if os.path.isfile(os.path.join(segments_dir, x)) and (x.endswith(current_v_ending) or x.endswith(manual_ending))]
        for srt_file in srt_files:

            # Skip the file if a manual override file of it exists
            if srt_file.endswith(current_v_ending) and srt_file.replace(current_v_ending, manual_ending) in srt_files:
                continue
            else:
                frames += count_frames_in_srt(BACKGROUND_LABEL, os.path.join(segments_dir, srt_file), rounding_ms)
    return frames

def count_total_label_ms(label: str, base_folder: str, rounding_ms: int) -> int:
    total_ms = 0
    segments_dir = os.path.join(base_folder, "segments")
    if os.path.isdir(segments_dir):
        srt_files = [x for x in os.listdir(segments_dir) if os.path.isfile(os.path.join(segments_dir, x)) and x.endswith(".v" + str(CURRENT_VERSION) + ".srt") or x.endswith(".MANUAL.srt")]
        for srt_file in srt_files:

            # Skip the file if a manual override file of it exists
            if srt_file.endswith(current_v_ending) and srt_file.replace(current_v_ending, manual_ending) in srt_files:
                continue
            else:
                total_ms += count_label_ms_in_srt(label, os.path.join(segments_dir, srt_file), rounding_ms)
    return total_ms

def count_label_ms_in_srt(label: str, srt_filename: str, rounding_ms: int) -> int:
    transition_events = parse_srt_file(srt_filename, rounding_ms, False)
    total_ms = 0
    start_ms = -1
    for transition_event in transition_events:
        if transition_event.label == label:
            start_ms = transition_event.start_ms
        elif start_ms > -1 and transition_event.label != label:
            total_ms += transition_event.start_ms - start_ms
            start_ms = -1
    
    return total_ms

def count_frames_in_srt(label: str, srt_filename: str, rounding_ms: int) -> int:
    transition_events = parse_srt_file(srt_filename, rounding_ms, False)
    start_ms = -1
    frames = 0
    for transition_event in transition_events:
        if transition_event.label == label:
            start_ms = transition_event.start_ms
        elif start_ms > -1 and transition_event.label != label:
            frames += round((transition_event.start_ms - start_ms - rounding_ms) / 15)
            start_ms = -1
    
    return frames

def print_detection_performance_compared_to_srt(actual_frames: List[DetectionFrame], frames_to_read: int, srt_file_location: str, output_wave_file = None):
    ms_per_frame = actual_frames[0].duration_ms
    transition_events = parse_srt_file(srt_file_location, ms_per_frame)
    detection_audio_frames = []
    total_ms = 0

    # Detection states
    detected_during_index = False
    false_detections = 0    
    
    # Times of recognitions
    total_occurrences = 0
    false_recognitions = 0
    positive_recognitions = 0
    total_recognitions = 0
    
    # Statistics
    ms_true_negative = 0
    ms_true_positive = 0
    ms_false_negative = 0
    ms_false_positive = 0
    
    false_types = {
        # Types of false negative recognitions
        "lag": [],
        "stutter": [],
        "cutoff": [],
        "full_miss": [],
        # Types of false positive recognitions        
        "late_stop": [],
        "missed_dip": [],
        "false_start": [],
        "full_false_positive": [],
    }
    
    # Loop over the results and compare them against the expected transition events
    index = 0
    t_index = 0
    for frame in actual_frames:
        index += 1
        total_ms += ms_per_frame

        # Determine expected label
        actual = frame.label
        expected = BACKGROUND_LABEL
        transitioning = False
        if t_index < len(transition_events):
            if t_index + 1 < len(transition_events) and index >= transition_events[t_index + 1].start_index:
                t_index += 1
                transitioning = True
                if transition_events[t_index].label != BACKGROUND_LABEL:
                    total_occurrences += 1
                # If the current label is a background label, we have just passed a full occurrence
                # So check if it has been found during the occurrence
                else:
                    if detected_during_index:
                        positive_recognitions += 1
                    else:
                        false_recognitions += 1
                detected_during_index = False
            expected = transition_events[t_index].label

        # Add a WAVE signal for each false and true positive detections
        if output_wave_file is not None:
            highest_amp = 65536 / 10
            signal_strength = highest_amp if actual != BACKGROUND_LABEL else 0
            if expected != actual and actual != BACKGROUND_LABEL:
                signal_strength = -highest_amp

            detection_signal = np.full(int(frames_to_read / 4), int(signal_strength))
            detection_signal[::2] = 0
            detection_signal[::3] = 0
            detection_signal[::5] = 0
            detection_signal[::7] = 0
            detection_signal[::9] = 0
            detection_audio_frames.append( detection_signal )
            
        if expected == actual:
            # Determine false detection types
            if false_detections > 0:
                false_index_start = index - false_detections
                false_index_end = index

                # Determine the amount of true events that have been miscategorized
                current_event_index = t_index
                first_index = t_index
                while( false_index_start < transition_events[first_index].start_index ):
                    first_index -= 1
                    if first_index <= 0:
                        first_index = 0
                        break
                        
                for ei in range(first_index - 1, current_event_index): 
                    event_index = ei + 1
                    event = transition_events[event_index]
                    event_start = event.start_index
                    event_end = transition_events[event_index + 1].start_index if event_index + 1 < len(transition_events) else len(actual_frames) - 1
                    
                    false_event_type = ""
                    ms_event = 0
                    if false_index_start <= event_start:
                        false_index_start = event_start

                        # Misrecognition of the start of an event
                        if false_index_end < event_end:
                            ms_event = (false_index_end - false_index_start ) * ms_per_frame
                            false_event_type = "late_stop" if event.label == BACKGROUND_LABEL else "lag"
                        # Misrecognition of a complete event
                        else:
                             ms_event = ( event_end - false_index_start ) * ms_per_frame
                             
                             false_event_type = "missed_dip" if event.label == BACKGROUND_LABEL else "full_miss"
                    elif false_index_start > event_start:
                    
                        # Misrecognition in between a full event
                        if false_index_end < event_end:
                            ms_event = ( false_index_end - false_index_start ) * ms_per_frame
                            false_event_type = "full_false_positive" if event.label == BACKGROUND_LABEL else "stutter"                            
                        # Misrecognition of the end of an event
                        else:
                            ms_event = (event_end - false_index_start) * ms_per_frame
                            false_event_type = "false_start" if event.label == BACKGROUND_LABEL else "cutoff"
                    
                    if false_event_type in false_types and ms_event > 0:
                        false_types[false_event_type].append( ms_event )

                    #if false_event_type == "full_miss":
                    #    print( "FULL MISS AT " + str(false_index_start * 15) )

                    # Reset the index to the start of the next event if the event can be followed by another false event
                    if false_event_type in ["false_start", "cutoff", "full_miss", "full_false_positive"]:
                        false_index_start = event_end

                false_detections = 0

            if expected != BACKGROUND_LABEL:
                if detected_during_index == False:
                    detected_during_index = True
                ms_true_positive += ms_per_frame
            else:
                ms_true_negative += ms_per_frame
        else:
            # False detections are counted by the sum of their events            
            false_detections += 1
    
    if output_wave_file is not None:
        output_wave_file.writeframes(b''.join(detection_audio_frames))
        output_wave_file.close()
    
    # Determine total time
    ms_false_positive = 0
    ms_false_negative = 0
    for false_type in false_types:
        false_types[false_type] = {
            "data": false_types[false_type],
        }

        amount = len(false_types[false_type]["data"])
        
        false_types[false_type]["times"] = amount            
        false_types[false_type]["avg"] = round(np.mean(false_types[false_type]["data"])) if amount > 0 else 0
        false_types[false_type]["std"] = round(np.std(false_types[false_type]["data"])) if amount > 0 else 0
        if false_type in ["late_stop", "missed_dip", "false_start", "full_false_positive"]:    
            ms_false_positive += round(np.sum(false_types[false_type]["data"]))
        else:
            ms_false_negative += round(np.sum(false_types[false_type]["data"]))
    
    # Export the results
    export_row = []
    print("-------- Detection statistics --------")
    print("Expected:                    " + str(total_occurrences) )
    export_row.append( str(positive_recognitions) )
    export_row.append( str(false_recognitions) )
    export_row.append( "0%" if total_occurrences == 0 else str(round(positive_recognitions / total_occurrences * 100)) + "%" )
    print("Found:                       "  + str(positive_recognitions) + " (" + ("0%" if total_occurrences == 0 else str(round(positive_recognitions / total_occurrences * 100)) + "%)") )
    print("Missed:                      " + str(false_recognitions) + " (" + ("0%" if total_occurrences == 0 else str(round(false_recognitions / total_occurrences * 100)) + "%)"))
    print("------------- Frame data -------------")
    print("Total frames:               " + str(len(actual_frames)))
    export_row.append( str(round((ms_true_positive + ms_true_negative) / total_ms * 1000) / 10) + "%" )
    print("Accuracy:                   " + export_row[-1])
    print("-------- Positive / negative --------")
    export_row.append( str(round(ms_true_positive / total_ms * 1000) / 10) + "%" )        
    print("True positive:              " + export_row[-1])
    export_row.append( str(round(ms_true_negative / total_ms * 1000) / 10) + "%" )
    print("True negative:              " + export_row[-1])
    export_row.append( str(round(ms_false_positive / total_ms * 1000) / 10) + "%" )
    print("False positive:             " + export_row[-1])
    export_row.append( str(round(ms_false_negative / total_ms * 1000) / 10) + "%" )
    print("False negative:             " + export_row[-1])
    print("----------- False positives ----------")
    key_length = 28
    if ms_false_positive > 0:
        for fp_type in [{"key": "false_start", "name": "Early start"},{"key": "missed_dip", "name": "Missed dip"},{"key": "late_stop", "name": "Late stop"},{"key": "full_false_positive", "name": "Full FP"},]:
            ms_total = sum(false_types[fp_type["key"]]["data"])
            print( (fp_type["name"] + " (% of FP):").ljust(key_length, " ") + ("0%" if ms_false_positive == 0 else str(round(ms_total / ms_false_positive * 100)) + "%") + " (" +  str(false_types[fp_type["key"]]["times"]) + "x)" )
            print("  [ Average " + str(false_types[fp_type["key"]]["avg"]) + "ms (σ " + str(false_types[fp_type["key"]]["std"]) + "ms) ]")
            export_row.append( str(false_types[fp_type["key"]]["times"]) )
            export_row.append( str(false_types[fp_type["key"]]["avg"]) + " σ " + str(false_types[fp_type["key"]]["std"]) if false_types[fp_type["key"]]["times"] > 0 else "0" )
    else:
        export_row.extend(["0", "0", "0", "0", "0", "0", "0", "0"])
    if ms_false_negative > 0:
        print("----------- False negatives ----------")
        for fn_type in [{"key": "lag", "name": "Lagged start"},{"key": "stutter", "name": "Stutter"},{"key": "cutoff", "name": "Early cut-off"},{"key": "full_miss", "name": "Full miss"},]:
            ms_total = sum(false_types[fn_type["key"]]["data"])
            print( (fn_type["name"] + " (% of FN):").ljust(key_length, " ") + ("0%" if ms_false_negative == 0 else str(round(ms_total / ms_false_negative * 100)) + "%") + " (" +  str(false_types[fn_type["key"]]["times"]) + "x)" )
            print("  [ Average " + str(false_types[fn_type["key"]]["avg"]) + "ms (σ " + str(false_types[fn_type["key"]]["std"]) + "ms) ]")
            export_row.append( str(false_types[fn_type["key"]]["times"]) )
            export_row.append( str(false_types[fn_type["key"]]["avg"]) + " σ " + str(false_types[fn_type["key"]]["std"]) if false_types[fn_type["key"]]["times"] > 0 else "0" )
    else:
        export_row.extend(["0", "0", "0", "0", "0", "0", "0", "0"])
    print("--------------------------------------")
    
    print("Excel row")
    print( ",".join(export_row) )