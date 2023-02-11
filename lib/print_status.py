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

def get_current_status(detection_state: DetectionState) -> List[str]:
    recorded_timestring = ms_to_srt_timestring( detection_state.ms_recorded, False)

    # Quality rating was manually established by doing some testing with added noise
    # And finding the results becoming worse when the SNR went lower than 10
    quality = ""
    if detection_state.ms_recorded > 10000:
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
        lines.append("| " + "Mic Quality: " + quality.rjust(LINE_LENGTH - 18) + " |")
    elif detection_state.state == "processing":
        lines.append("| " + "PROCESSING...".ljust(LINE_LENGTH) + " |")
    elif detection_state.state == "paused":
        lines.append("| " + "PAUSED - Resume using SPACE".ljust(LINE_LENGTH) + " |")
    else:
        lines.append("| " + detection_state.state.upper().ljust(LINE_LENGTH) + " |")
    
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
        quantity = ""
        if label.ms_detected < 16500:
            quantity = "Not enough"
        elif label.ms_detected > 16500 and label.ms_detected < 41250:
            quantity = "Sufficient"
        elif label.ms_detected >= 41250 and label.ms_detected < 82500:
            quantity = "Good"
        elif label.ms_detected >= 82500:
            quantity = "Excellent"

        lines.extend([
           "|".ljust(LINE_LENGTH - 2,"-") + "|",
            "| " + label.label.ljust(LINE_LENGTH - 5) + " |",
            "| " + "Recorded: " + ms_to_srt_timestring( label.ms_detected, False ).rjust(LINE_LENGTH - 15) + " |",
            "| " + "Data Quantity: " + quantity.rjust(LINE_LENGTH - 20) + " |",
        ])
        
        if detection_state.advanced_logging:
            lines.append( "| " + ("type:" + str(label.duration_type if label.duration_type else "Unknown").upper().rjust(LINE_LENGTH - 10)) + " |" )
            lines.append( "| " + ("dBFS treshold:" + str(round(label.min_dBFS, 2)).rjust(LINE_LENGTH - 19)) + " |" )
    lines.append("'".ljust(LINE_LENGTH - 2,"-") + "'")
    
    return lines

def clear_previous_lines(line_count):
    line = "";
    for i in range(0,line_count):
        line += ANSI_CODE_LINE_UP
    print(line, end=ANSI_CODE_LINE_CLEAR )