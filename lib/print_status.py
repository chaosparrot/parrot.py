from .typing import DetectionState
from typing import List
from .srt import ms_to_srt_timestring
import os
import sys

# Needed to make escape characters work on Windows for some reason
if os.name == 'nt':
    os.system("")
ANSI_CODE_LINE_UP = '\033[1A'
ANSI_CODE_LINE_CLEAR = '\x1b[2K'

# If no UTF-8 characters are supported, use ascii characters instead
PROGRESS_FILLED = '#' if sys.stdout.encoding != 'utf-8' else '\u2588'
PROGRESS_AVAILABLE = '-' if sys.stdout.encoding != 'utf-8' else '\u2591'
LINE_LENGTH = 50

def create_progress_bar(percentage: float = 1.0) -> str:
    filled_characters = round(max(0, min(LINE_LENGTH, LINE_LENGTH * percentage)))
    return "".rjust(filled_characters, PROGRESS_FILLED).ljust(LINE_LENGTH, PROGRESS_AVAILABLE)

def get_current_status(detection_state: DetectionState, extra_states: List[DetectionState] = []) -> List[str]:
    total_ms_recorded = detection_state.ms_recorded
    for extra_state in extra_states:
        total_ms_recorded += extra_state.ms_recorded
    recorded_timestring = ms_to_srt_timestring( total_ms_recorded, False)

    # Quality rating was manually established by doing some testing with added noise
    # And finding the results becoming worse when the SNR went lower than 10
    quality = ""
    if total_ms_recorded > 10000:
        if detection_state.expected_snr >= 25:
            quality = "Excellent"
        elif detection_state.expected_snr >= 20:
            quality = "Great"
        elif detection_state.expected_snr >= 15:
            quality = "Good"
        elif detection_state.expected_snr >= 10:
            quality = "Average"
        elif detection_state.expected_snr >= 7:
            quality = "Poor"
        else:
            quality = "Unusable"
        
    lines = [
        ".".ljust(LINE_LENGTH - 2, "-") + ".",
        "| " + "Listening for:" + recorded_timestring.rjust(LINE_LENGTH - 19) + " |",
    ]
    
    if detection_state.state == "recording":
        if detection_state.latest_dBFS <= -100:
            lines.append("| " + "WEAK SIGNAL - Please unmute microphone".ljust(LINE_LENGTH - 5) + " |")        
        else:
            lines.append("| " + "Sound Quality: " + quality.rjust(LINE_LENGTH - 20) + " |")
    elif detection_state.state == "processing":
        lines.append("| " + "PROCESSING...".ljust(LINE_LENGTH - 5) + " |")
    elif detection_state.state == "paused":
        lines.append("| " + "PAUSED - Resume using SPACE".ljust(LINE_LENGTH - 5) + " |")
    else:
        lines.append("| " + detection_state.state.upper().ljust(LINE_LENGTH - 5) + " |")
    
    lines.append("| " + ("dBFS:" + str(round(detection_state.latest_dBFS)).rjust(LINE_LENGTH - 10)) + " |")
    lines.append("| " + ("Δ:" + str(round(detection_state.latest_delta)).rjust(LINE_LENGTH - 7)) + " |")
    if detection_state.advanced_logging:
       lines.extend([
           "|".ljust(LINE_LENGTH - 2,"-") + "|",
           "| " + "Est. values for thresholding".ljust(LINE_LENGTH - 5) + " |",
           "|".ljust(LINE_LENGTH - 2,"-") + "|",
           "| " + ("Noise floor (dBFS):" + str(round(detection_state.expected_noise_floor)).rjust(LINE_LENGTH - 24)) + " |",
           "| " + ("SNR:" + str(round(detection_state.expected_snr)).rjust(LINE_LENGTH - 9)) + " |",
       ])

    for label in detection_state.labels:
        # Quantity rating is based on 5000 30ms windows being good enough to train a label from the example model
        # And 1000 30ms windows being enough to train a label decently
        # With atleast 10 percent extra for a possible hold-out set during training
        total_ms_detected = label.ms_detected + label.previous_detected
        for extra_state in extra_states:
            for extra_label in extra_state.labels:
                if extra_label.label == label.label:
                    total_ms_detected += extra_label.ms_detected + extra_label.previous_detected
        
        percent_to_next = 0
        quantity = ""
        if total_ms_detected < 16500:
            percent_to_next = (total_ms_detected / 16500 ) * 100
            quantity = "Not enough"
        elif total_ms_detected > 16500 and total_ms_detected < 41250:
            percent_to_next = ((total_ms_detected - 16500) / (41250 - 16500) ) * 100
            quantity = "Sufficient"
        elif total_ms_detected >= 41250 and total_ms_detected < 82500:
            percent_to_next = ((total_ms_detected - 41250) / (82500 - 41250) ) * 100        
            quantity = "Good"
        elif total_ms_detected >= 82500:
            quantity = "Excellent"
            
        if percent_to_next != 0:
            quantity += " (" + str(round(percent_to_next)) + "%)"

        lines.extend([
           "|".ljust(LINE_LENGTH - 2,"-") + "|",
            "| " + label.label.ljust(LINE_LENGTH - 5) + " |",
            "| " + "Recorded: " + ms_to_srt_timestring( total_ms_detected, False ).rjust(LINE_LENGTH - 15) + " |",
            "| " + "Data Quantity: " + quantity.rjust(LINE_LENGTH - 20) + " |",
        ])
        
        if detection_state.advanced_logging:
            lines.append( "| " + ("type:" + str(label.duration_type if label.duration_type else "DETERMINING...").upper().rjust(LINE_LENGTH - 10)) + " |" )
            lines.append( "| " + ("dBFS treshold:" + str(round(label.min_dBFS, 2)).rjust(LINE_LENGTH - 19)) + " |" )
    lines.append("'".ljust(LINE_LENGTH - 2,"-") + "'")
    
    return lines

def reset_previous_lines(line_count):
    line = "";
    for i in range(0,line_count):
        line += ANSI_CODE_LINE_UP
    print(line, end=ANSI_CODE_LINE_CLEAR )

def clear_previous_lines(line_count):
    for i in range(0,line_count):
        print(ANSI_CODE_LINE_UP, end=ANSI_CODE_LINE_CLEAR )