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
    euclid_dist: float
    mel_data: List[List[float]]
    label: str

@dataclass
class DetectionEvent:
    label: str
    
    # Based on wave indecis
    start_index: int
    end_index: int
    start_ms: int
    end_ms: int
    frames: List[DetectionFrame]

@dataclass
class DetectionLabel:
    label: str
    ms_detected: int
    previous_detected: int
    duration_type: str
    
    min_ms: float
    min_dBFS: float
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
    expected_snr: float
    expected_noise_floor: float
    labels: List[DetectionLabel]
    override_labels: List[DetectionLabel] = None