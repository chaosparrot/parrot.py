from .typing import DetectionLabel, DetectionFrame, DetectionEvent, DetectionState
from config.config import BACKGROUND_LABEL
from typing import List

CURRENT_VERSION = 1

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
