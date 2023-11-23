from dataclasses import dataclass
from typing import List

@dataclass
class TransitionEvent:
    label: str
    start_index: int
    start_ms: int

@dataclass
class DetectionFrame:
    index: int
    duration_ms: int
    positive: bool
    power: float
    dBFS: float
    filtered_dBFS: float
    zero_crossing: int
    euclid_dist: float
    mel_data: List[List[float]]
    label: str
    previous_frame = None

@dataclass
class DetectionEvent:
    label: str
    
    # Based on wave indecis
    start_index: int
    end_index: int
    start_ms: int
    end_ms: int
    average_dBFS: float
    average_mel_data: List[List[float]]    
    frames: List[DetectionFrame]

@dataclass
class DetectionLabel:
    label: str
    ms_detected: int
    previous_detected: int
    duration_type: str
    
    min_ms: float
    min_dBFS: float
    min_secondary_dBFS: float
    min_distance: float
    max_distance: float

@dataclass
class DetectionState:
    strategy: str
    state: str
    ms_per_frame: int
    ms_recorded: int
    advanced_logging: bool

    latest_dBFS: float
    latest_delta: float    
    
    expected_snr: float
    expected_noise_floor: float
    labels: List[DetectionLabel]
    override_labels: List[DetectionLabel] = None
    current_dBFS_threshold: float = None
    current_zero_crossing_threshold = None