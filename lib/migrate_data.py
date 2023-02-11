from config.config import *
import os
from lib.stream_processing import CURRENT_VERSION, process_wav_file
from lib.print_status import create_progress_bar, clear_previous_lines, get_current_status
from .typing import DetectionState
import time

def check_migration():
    version_detected = CURRENT_VERSION
    recording_dirs = os.listdir(RECORDINGS_FOLDER)
    for file in recording_dirs:
        if os.path.isdir(os.path.join(RECORDINGS_FOLDER, file)):
            if not os.path.exists(os.path.join(RECORDINGS_FOLDER, file, "segments")) \
                or not os.listdir(os.path.join(RECORDINGS_FOLDER, file, "segments")):
                version_detected = 0

    if version_detected < CURRENT_VERSION:
        print("----------------------------")
        print("!! Improvement to segmentation found !!")
        print("This can help improve the data gathering from your recordings which make newer models better")
        update = input("Do you want to reprocess your recordings? [y/N] ")
        if (update.lower() == "y"):
            migrate_data()

def migrate_data():
    print("----------------------------")
    recording_dirs = os.listdir(RECORDINGS_FOLDER)
    for label in recording_dirs:
        source_dir = os.path.join(RECORDINGS_FOLDER, label, "source")
        if os.path.isdir(source_dir):
            segments_dir = os.path.join(RECORDINGS_FOLDER, label, "segments")
            if not os.path.exists(segments_dir):
                os.makedirs(segments_dir)
            print( "Resegmenting " + label + "..." )
            print( "" )
            wav_files = [x for x in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, x)) and x.endswith(".wav")]            
            progress = 0
            progress_chunk = 1 / len( wav_files )
            for index, wav_file in enumerate(wav_files):
                wav_file_location = os.path.join(source_dir, wav_file)            
                srt_file_location = os.path.join(segments_dir, wav_file.replace(".wav", ".v" + str(CURRENT_VERSION) + ".srt"))
                output_file_location = os.path.join(segments_dir, wav_file.replace(".wav", "_detection.wav"))

                process_wav_file(wav_file_location, srt_file_location, output_file_location, [label], \
                    lambda internal_progress, state: print_migration_progress(progress + (internal_progress * progress_chunk), state) )
                progress = index / len( wav_files ) + progress_chunk
                clear_previous_lines(1)
                print( create_progress_bar(progress) )
            clear_previous_lines(1)
            clear_previous_lines(1)
            print( label + " updated!" )

    time.sleep(1)

def print_migration_progress(progress, state: DetectionState):
    status_lines = get_current_status(state)
    line_count = 1 + len(status_lines) if state.ms_recorded > 0 else 1
    clear_previous_lines(line_count)
    print( create_progress_bar(progress) )
    if progress < 1:
        for line in status_lines:
            print( line )
    