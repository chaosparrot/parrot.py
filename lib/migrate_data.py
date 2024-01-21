from config.config import *
import os
from lib.stream_processing import process_wav_file
from lib.print_status import create_progress_bar, clear_previous_lines, get_current_status, reset_previous_lines
from .typing import DetectionState
import time

def check_migration():
    version_detected = CURRENT_VERSION
    recording_dirs = os.listdir(RECORDINGS_FOLDER)
    for file in recording_dirs:
        if file != BACKGROUND_LABEL and os.path.isdir(os.path.join(RECORDINGS_FOLDER, file)):
            segments_folder = os.path.join(RECORDINGS_FOLDER, file, "segments")
            if not os.path.exists(segments_folder):
                version_detected = 0
                break
            else:
                source_files = [x for x in os.listdir(os.path.join(RECORDINGS_FOLDER, file, "source")) if x.endswith(".wav")]
                for source_file in source_files:
                    srt_file = source_file.replace(".wav", ".v" + str(CURRENT_VERSION) + ".srt")
                    thresholds_file_location = os.path.join(segments_folder, source_file.replace(".wav", "_thresholds.txt"))
                    
                    manual_srt_file = source_file.replace(".wav", ".MANUAL.srt")
                    if not os.path.exists(os.path.join(segments_folder, srt_file)) and not os.path.exists(os.path.join(segments_folder, manual_srt_file)):
                        version_detected = 0
                        break

                    # If an override file exists and the time of modification is later than the manual SRT file generated, we need to resegment
                    elif os.path.exists(thresholds_file_location):
                        srt_file_location = os.path.join(segments_folder, srt_file)
                        manual_srt_file_location = os.path.join(segments_folder, manual_srt_file)
                        if os.path.exists(manual_srt_file_location):
                            if os.path.getmtime(thresholds_file_location) > os.path.getmtime(manual_srt_file_location):
                                version_detected = 0

                        elif os.path.exists(srt_file_location):
                            # Thresholds file has been changed manually - We need to resegment
                            if os.path.getmtime(srt_file_location) + 5 < os.path.getmtime(thresholds_file_location):
                                version_detected = 0

                
                if version_detected == 0:
                    break
    
    if version_detected < CURRENT_VERSION:
        print("----------------------------")
        print("!! Improvement to segmentation found !!")
        print("This can help improve the data gathering from your recordings which make newer models better")
        print("Resegmenting your data may take a while")
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
            wav_files = [x for x in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, x)) and x.endswith(".wav")]            
            if len(wav_files) == 0:
                continue
            print( "Resegmenting " + label + "..." )
            previous_progress = 0
            progress = 0
            progress_chunk = 1 / len( wav_files )
            file_progress = 0
            next_file_progress = progress_chunk
            skipped_amount = 0
            files_to_resegment = []
            for index, wav_file in enumerate(wav_files):
                wav_file_location = os.path.join(source_dir, wav_file)
                srt_file_location = os.path.join(segments_dir, wav_file.replace(".wav", ".v" + str(CURRENT_VERSION) + ".srt"))
                output_file_location = os.path.join(segments_dir, wav_file.replace(".wav", "_comparison.wav"))
                thresholds_file_location = os.path.join(segments_dir, wav_file.replace(".wav", "_thresholds.txt"))
                override_file_location = None
                should_resegment_file = not os.path.exists(srt_file_location)                
                
                # Make sure that thresholds overrides get a different SRT file postfix
                if os.path.exists(thresholds_file_location):
                    thresholds_file_modification_time = os.path.getmtime(thresholds_file_location)
                    is_manual_threshold = False
                    manual_srt_file_location = os.path.join(segments_dir, wav_file.replace(".wav", ".MANUAL.srt"))

                    # If the thresholds file has been changed after the SRT file has been created, we need to resegment the file as a manual change
                    if ( os.path.exists(manual_srt_file_location) ) or ( os.path.exists(srt_file_location) and os.path.getmtime(srt_file_location) + 5 < thresholds_file_modification_time ):
                        srt_file_location = manual_srt_file_location
                        is_manual_threshold = True

                    should_resegment_file = not os.path.exists(srt_file_location) or os.path.getmtime(srt_file_location) + 5 < thresholds_file_modification_time

                    # Make sure we do not override the thresholds file if it has been changed manually                    
                    if is_manual_threshold:
                        override_file_location = thresholds_file_location
                        thresholds_file_location = None

                # Only resegment if the new version does not exist already
                if should_resegment_file:
                    files_to_resegment.append([wav_file_location, srt_file_location, output_file_location, thresholds_file_location, override_file_location])
                else:
                    skipped_amount += 1

            # Calculate the progress of the files we need to resegment
            progress_chunk = 1 / len(wav_files)
            base_progress = skipped_amount * progress_chunk
            file_progress = base_progress
            progress = file_progress
            next_file_progress = file_progress + progress_chunk
            for index, data in enumerate(files_to_resegment):
                wav_file_location = data[0]
                srt_file_location = data[1]
                output_file_location = data[2]
                thresholds_file_location = data[3]
                override_file_location = data[4]

                process_wav_file(wav_file_location, srt_file_location, output_file_location, thresholds_file_location, [label], \
                    lambda internal_progress, state: print_migration_progress(progress + (internal_progress * progress_chunk), state, file_progress, next_file_progress), None, override_file_location )
                
                file_progress += progress_chunk
                next_file_progress += progress_chunk
                progress = file_progress

            clear_previous_lines(1)
            print( label + " resegmented!" if skipped_amount < len(wav_files) else label + " already properly segmented!" )

    time.sleep(1)
    print("Finished migrating data!")
    print("----------------------------")

def print_migration_progress(progress, state: DetectionState, base_progress, next_base_progress):
    status_lines = get_current_status(state)
    status_lines.insert(0, create_progress_bar(progress))
    line_count = len(status_lines) if progress - base_progress > 0 or state.state == "processing" else 0
    finished_loading_file = next_base_progress - progress == 0

    # Only do a full clear when the progress has finished
    # Or when it is transitioning to post processing
    if finished_loading_file or ( state.state == "processing" and progress - base_progress == 0 ):
        clear_previous_lines(line_count)

    # Only start rewriting lines when we have printed the first element
    elif progress - base_progress > 0:
        reset_previous_lines(line_count)

    if not finished_loading_file:
        for line in status_lines:
            print( line )