from config.config import *
import os
from lib.stream_processing import CURRENT_VERSION
from lib.print_status import create_progress_bar, clear_previous_lines
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
    for file in recording_dirs:
        source_dir = os.path.join(RECORDINGS_FOLDER, file, "source")
        if os.path.isdir(source_dir):
            segments_dir = os.path.join(RECORDINGS_FOLDER, file, "segments")
            if not os.path.exists(segments_dir):
                os.makedirs(segments_dir)
            print( "Resegmenting " + file + "..." )
            wav_files = [x for x in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, x)) and x.endswith(".wav")]            
            progress = 0
            progress_chunk = 1 / len( wav_files )
            print( create_progress_bar(progress) )
            for index, wav_file in enumerate(wav_files):
                srt_file = os.path.join(segments_dir, wav_file.replace(".wav", ".v1.srt"))
                
                progress = index / len( wav_files ) + progress_chunk
                clear_previous_lines(1)
                print( create_progress_bar(progress) )
            clear_previous_lines(1)
            clear_previous_lines(1)            
            print( file + " updated!" )            

    time.sleep(1)    
    