from config.config import *
import numpy as np
import math
import random
import time
import os
import warnings
from lib.machinelearning import *
from lib.srt import count_total_frames, count_total_silence_frames
from lib.wav import load_wav_files_with_srts, load_wav_data_from_srt, load_sequential_wav_data_from_srt

def get_grouped_data_directories( labels ):
    # If the microphone separator setting is set use that to split directory names into categories/labels.
    # This enable us to have multiple directories with different names and as long as they have the same prefix they will be combined into a single category/label.
    grouped_data_directories = {}
    for directory_name in labels:
        if MICROPHONE_SEPARATOR:
            category_name = directory_name.split( MICROPHONE_SEPARATOR )[0] 
        else:
            category_name = directory_name
        if category_name not in grouped_data_directories:
            grouped_data_directories[ category_name ] = []
        data_directory = f"{ DATASET_FOLDER }/{ directory_name }"
        grouped_data_directories[ category_name ].append( data_directory )
    return grouped_data_directories

def generate_data_balance_strategy_map(grouped_data_directories):
    ms_per_frame = math.floor(RECORD_SECONDS / SLIDING_WINDOW_AMOUNT * 1000)
    directory_counts = {}
    max_size = 0
    min_size = 2147483647 # Large number so that minimum size is always smaller
    background_label_size = 0
    for index, label in enumerate( grouped_data_directories ):
        directories = grouped_data_directories[ label ]
        label_count = 0
        for directory in directories:
            label_count += count_total_frames(label, directory, ms_per_frame)
            background_label_size += count_total_silence_frames(directory, ms_per_frame)
        directory_counts[label] = label_count
        max_size = max(max_size, label_count)
        if label_count > 0:
            min_size = min(min_size, label_count)

    strategies = ['oversample', 'undersample', 'sample', 'background']
    max_oversample_ratio = 2
    
    std = np.std(list(directory_counts.values()))
    average = np.mean(list(directory_counts.values()))
    total_truncation = math.floor(average + std / 2)
    
    sampling_strategies = {}
    for label in directory_counts:            
        strategy = strategies[2]
        total_loaded = directory_counts[label]
        if AUTOMATIC_DATASET_BALANCING:
            if directory_counts[label] < total_truncation / 1.25:
                strategy = strategies[0]
                total_loaded = min(directory_counts[label] * max_oversample_ratio, total_truncation)
            elif directory_counts[label] > total_truncation * 1.25:
                strategy = strategies[1]
                total_loaded = total_truncation

        sampling_strategies[label] = {
            "strategy": strategy,
            "total_size": directory_counts[label],
            "total_loaded": total_loaded,
            "truncate_after": total_truncation,
            "sample_from_each": -1
        }

    #sampling_strategies[BACKGROUND_LABEL] = {
    #    "strategy": strategies[3],
    #    "truncate_after": total_truncation,
    #    "total_loaded": total_truncation,
    #    "total_size": background_label_size,
    #    "sample_from_each": round(min(total_truncation, background_label_size) / len(grouped_data_directories.keys()))
    #}

    return rebalance_sampling_strategies_for_memory(sampling_strategies)

def rebalance_sampling_strategies_for_memory(sampling_strategies):
    if not SHOULD_FIT_INSIDE_RAM or not AUTOMATIC_DATASET_BALANCING:
        return sampling_strategies

    # Make sure the additional data loaded does not increase past a certain point
    total_data_size = 0
    total_truncation = 0
    for label in sampling_strategies:
        total_data_size += sampling_strategies[label]["total_loaded"]
        total_truncation = sampling_strategies[label]["truncate_after"]
    
    strategies = ['oversample', 'undersample', 'sample', 'background']
    max_oversample_ratio = 2
    
    # Rebalance the data if it doesn't fit inside of RAM completely
    avg_sample_size = 160 * 4 # Based on 30 ms mel cepstrum encoded as float32
    avg_sample_size += 32 # And an overhead of an int and a string reference
    max_ram = MAX_RAM
    ram_used_for_data = total_data_size * avg_sample_size
    reduced = 0
    new_data_size = 0
    if ram_used_for_data > max_ram:
        print ("Rebalancing data to accomodate max size in RAM..." )
        print ("Assumed max RAM of " + str(max_ram / 1000000000) + "GB, est. data size in RAM: " + str(ram_used_for_data / 1000000000) + "GB")
        total_truncation = math.floor(total_truncation * (max_ram / ram_used_for_data))

        for label in sampling_strategies:
            strategy = strategies[2]
            total_loaded = sampling_strategies[label]['total_size']
            if total_loaded < total_truncation / 1.25:
                strategy = strategies[0]
                total_loaded = min(total_loaded * max_oversample_ratio, total_truncation)
            elif total_loaded > total_truncation * 1.25:
                strategy = strategies[1]
                total_loaded = total_truncation

            sampling_strategies[label]["truncate_after"] = total_truncation
            sampling_strategies[label]["total_loaded"] = total_loaded

            # Rebalance the samples taken for silence as well
            if sampling_strategies[label]["strategy"] == "background":
                sampling_strategies[label]["sample_from_each"] = round(min(total_truncation, sampling_strategies[label]['total_size']) / len(sampling_strategies.keys()))
            new_data_size += total_loaded
        print( "Reduced data by ~" + str(math.ceil(100 - (new_data_size / total_data_size * 100))) + "% to fit the whole dataset inside RAM") 
    
    return sampling_strategies

def get_listed_files_from_label(label, grouped_data_directories):
    warnings.filterwarnings("ignore", "n_fft=2048 is too small for input signal")
    directories = grouped_data_directories[ label ]

    listed_files = {}
    for directory in directories:
        segments_directory = os.path.join(directory, "segments")
        source_directory = os.path.join(directory, "source")                
        if not (os.path.exists(segments_directory) and os.path.exists(source_directory)):
            continue
        
        source_files = os.listdir(source_directory)
        srt_files = [x for x in os.listdir(segments_directory) if x.endswith(".srt")]
        for source_file in source_files:
            shared_key = source_file.replace(".wav", "")
            
            possible_srt_files = [x for x in srt_files if x.startswith(shared_key)]
            if len(possible_srt_files) == 0:
                continue
                
            # Find the highest version of the segmentation for this source file
            srt_file = possible_srt_files[0]
            for possible_srt_file in possible_srt_files:
                if srt_file.endswith(".MANUAL.srt"):
                    break
                
                if possible_srt_file.endswith(".MANUAL.srt"):
                    srt_file = possible_srt_file
                else:                
                    current_version = int( srt_file.replace(".srt", "").replace(shared_key + ".v", "") )
                    version = int( possible_srt_file.replace(".srt", "").replace(shared_key + ".v", "") )
                    if version > current_version:
                        srt_file = possible_srt_file
            
            listed_files[os.path.join(source_directory, source_file)] = os.path.join(segments_directory, srt_file)
    return listed_files

def sample_data_from_label(label, grouped_data_directories, sample_strategies, input_type):
    listed_files = get_listed_files_from_label(label, grouped_data_directories)
    listed_files_size = len( listed_files )
    
    data = {"background": [], "background_augmented": [], "label": [], "augmented": []}
    total_label_samples = []
    total_augmented_samples = []
    total_background_samples = []
    total_augmented_background_samples = []
    
    if label in sample_strategies:
        strategy = sample_strategies[label]["strategy"]
        truncate_after = sample_strategies[label]["truncate_after"]
        if strategy == "oversample":
            print( f"Loading in {label} using oversampling: +" + str(abs(round(sample_strategies[label]["total_loaded"] / sample_strategies[label]["total_size"] * 100) - 100)) + "%" )
        elif strategy == "undersample":
            print( f"Loading in {label} using undersampling: -" + str(abs(round(sample_strategies[label]["total_loaded"] / sample_strategies[label]["total_size"] * 100) - 100)) + "%" )
        elif strategy == "background":
            print( f"Loading in {label} by sampling from other labels" )
            
            # Early return for background loading as we do that during other loading sequences
            return data
        else:
            print( f"Loading in {label}" )

        should_oversample = strategy == "oversample"

        listed_source_files = listed_files.keys()
        for file_index, full_filename in enumerate( listed_source_files ):
            label_samples = load_wav_data_from_srt(listed_files[full_filename], full_filename, input_type, should_oversample)
            for sample in label_samples:
                total_label_samples.append([full_filename, sample])

            augmented_samples = load_wav_data_from_srt(listed_files[full_filename], full_filename, input_type, should_oversample, True)
            for augmented_sample in augmented_samples:
                total_augmented_samples.append([full_filename, augmented_sample])

            background_samples = load_wav_data_from_srt(listed_files[full_filename], full_filename, input_type, False, False, True)
            for background_sample in background_samples:
                total_background_samples.append([full_filename, background_sample])
            
            augmented_background_samples = load_wav_data_from_srt(listed_files[full_filename], full_filename, input_type, True, False, True)
            for augmented_background_sample in augmented_background_samples:
                total_augmented_background_samples.append([full_filename, augmented_background_sample])

        seed = round(time.time() * 1000)

        # Truncate the sample data randomly, but ensure the seed is the same so that the augmented data matches the non-augmented data index
        if strategy in ["oversample", "undersample"] and len(total_label_samples) > truncate_after:
            random.seed(seed)
            total_label_samples = random.sample(total_label_samples, truncate_after)
            random.seed(seed)
            total_augmented_samples = random.sample(total_augmented_samples, truncate_after)

    data["label"] = total_label_samples
    data["augmented"] = total_augmented_samples
    data["background"] = total_background_samples
    data["background_augmented"] = total_augmented_background_samples
    return data

def shannon_entropy(label_counts):
    totals = list(label_counts.values())
    n = sum(totals)
    h = -sum([(count / n) * np.log((count / n)) for count in totals])
    return h / np.log(len(totals))

def load_sklearn_data( filtered_data_directory_names, input_type ):
    grouped_data_directories = get_grouped_data_directories( filtered_data_directory_names )
    sample_strategies = generate_data_balance_strategy_map(grouped_data_directories )
    
    dataset = {}
    dataset[BACKGROUND_LABEL] = []
    for label in grouped_data_directories:
        if label != BACKGROUND_LABEL:
            data_sample = sample_data_from_label( label, grouped_data_directories, sample_strategies, input_type)
            dataset[label] = [x[1] for x in data_sample["label"]]
            dataset[label].extend([x[1] for x in data_sample["augmented"]])
            dataset[BACKGROUND_LABEL].extend([x[1] for x in data_sample["background"]])
            dataset[BACKGROUND_LABEL].extend([x[1] for x in data_sample["background_augmented"]])

    # Generate the training set and labels with them
    dataset_x = []
    dataset_labels = []

    for label, data in dataset.items():
        # Add a label used for classifying the sounds
        dataset_x.extend( data )
        dataset_labels.extend([label for x in data])

    return dataset_x, dataset_labels, grouped_data_directories.keys()
    
def load_pytorch_data( filtered_data_directory_names, input_type):
    import torch

    grouped_data_directories = get_grouped_data_directories( filtered_data_directory_names )
    sample_strategies = generate_data_balance_strategy_map(grouped_data_directories )

    dataset = {}
    augmented = {}
    dataset[BACKGROUND_LABEL] = []
    augmented[BACKGROUND_LABEL] = []
    for label in grouped_data_directories:
        if label != BACKGROUND_LABEL:
            data_sample = sample_data_from_label( label, grouped_data_directories, sample_strategies, input_type)
            dataset[label] = [[x[0], torch.tensor(x[1]).float()] for x in data_sample["label"]]
            augmented[label] =[[x[0], torch.tensor(x[1]).float()] for x in data_sample["augmented"]]
            dataset[BACKGROUND_LABEL].extend([[x[0], torch.tensor(x[1]).float()] for x in data_sample["background"]])
            augmented[BACKGROUND_LABEL].extend([[x[0], torch.tensor(x[1]).float()] for x in data_sample["background_augmented"]])
    
    return {
        "data": dataset,
        "augmented": augmented
    }

def load_sequential_pytorch_data( filtered_data_directory_names, input_type, sequence_length = 3):
    import torch

    grouped_data_directories = get_grouped_data_directories( filtered_data_directory_names )
    sample_strategies = generate_data_balance_strategy_map(grouped_data_directories )

    dataset = {"__streams": []}
    augmented = {"__streams": []}
    dataset[BACKGROUND_LABEL] = []
    augmented[BACKGROUND_LABEL] = []
    for label in grouped_data_directories:
        if label not in dataset:
            dataset[label] = []
            augmented[label] = []

        if label != BACKGROUND_LABEL:
            data_samples = sample_sequential_data_from_label( label, grouped_data_directories, sample_strategies, input_type, sequence_length=sequence_length)

            # Add the data stream
            stream_index_offset = len(dataset["__streams"])
            for stream in data_samples["streams"]:
                torch_stream = []
                for stream_frame in stream:
                    torch_stream.append(torch.tensor(stream_frame[0]).float())
                dataset["__streams"].append(torch_stream)

            # Add the augmented stream
            for augmented_stream in data_samples["augmented_streams"]:
                torch_stream = []
                for stream_frame in augmented_stream:
                    torch_stream.append(torch.tensor(stream_frame[0]).float())
                augmented["__streams"].append(torch_stream)

            # Add the indices on the stream for each label
            for frame_sequence in data_samples["label"]:
                dataset[label].append([[stream_index_offset + x[0], x[1], x[2]] for x in frame_sequence])
                augmented[label].append([[stream_index_offset + x[0], x[1], x[2]] for x in frame_sequence])

            for frame_sequence in data_samples["background"]:
                dataset[BACKGROUND_LABEL].append([[stream_index_offset + x[0], x[1], x[2]] for x in frame_sequence])
                augmented[BACKGROUND_LABEL].append([[stream_index_offset + x[0], x[1], x[2]] for x in frame_sequence])

    return {
        "data": dataset,
        "augmented": augmented
    }

def sample_sequential_data_from_label(label, grouped_data_directories, sample_strategies, input_type, sequence_length=1):
    listed_files = get_listed_files_from_label(label, grouped_data_directories)

    # TODO - IMPROVE DATA LOADING SPEED? Currently takes twice as long

    data = {"background": [], "background_augmented": [], "label": [], "augmented": []}

    if label in sample_strategies:
        strategy = sample_strategies[label]["strategy"]
        truncate_after = sample_strategies[label]["truncate_after"]
        if strategy == "oversample":
            print( f"Loading in {label} using oversampling: +" + str(abs(round(sample_strategies[label]["total_loaded"] / sample_strategies[label]["total_size"] * 100) - 100)) + "%" )
        elif strategy == "undersample":
            print( f"Loading in {label} using undersampling: -" + str(abs(round(sample_strategies[label]["total_loaded"] / sample_strategies[label]["total_size"] * 100) - 100)) + "%" )
        elif strategy == "background":
            print( f"Loading in {label} by sampling from other labels" )

            # Early return for background loading as we do that during other loading sequences
            return data
        else:
            print( f"Loading in {label}" )

        should_oversample = strategy == "oversample"

        data["streams"] = []
        data["augmented_streams"] = []

        listed_source_files = listed_files.keys()
        total_count_label_samples = 0
        for file_index, full_filename in enumerate( listed_source_files ):
            all_sample_streams = load_sequential_wav_data_from_srt(listed_files[full_filename], full_filename, input_type, should_oversample)
            augmented_sample_streams = load_sequential_wav_data_from_srt(listed_files[full_filename], full_filename, input_type, should_oversample, True)

            data["streams"].extend(all_sample_streams)
            data["augmented_streams"].extend(augmented_sample_streams)
        
        # Collect the label events and count
        # The total label samples for over- and undersampling
        total_label_events = []
        for stream_index, stream in enumerate(data["streams"]):
            current_label_event = []            
            for stream_frame_index, stream_frame in enumerate(stream):
                if stream_frame[1] == label:
                    if stream_frame_index > sequence_length - 1:
                        total_count_label_samples += 1
                    current_label_event.append([stream_index, stream_frame_index])
                else:
                    if len(current_label_event) > 0:
                        total_label_events.append(current_label_event)
                    current_label_event = []

            if len(current_label_event) > 0:
                total_label_events.append(current_label_event)

        seed = round(time.time() * 1000)

        # Remove events 
        if strategy in ["oversample", "undersample"] and total_count_label_samples > truncate_after:
            # Shuffle the total events randomly so our later removal is randomly sampled
            random.shuffle(total_label_events)

            # Remove events one by one until the total count is below the truncated threshold
            removed_events = []
            while total_count_label_samples > truncate_after:
                removed_event = total_label_events.pop()
                removed_events.append(removed_event)
                total_count_label_samples -= len(removed_event)

            # Sort the events chronologically again so we can easily split streams
            removed_events.sort(key = lambda event: (event[0][0] * 1000000) + event[0][1])

            # Rebuild the streams and augmented streams by splitting up the existing ones
            new_streams = []
            new_augmented_streams = []
            for stream_index, stream in enumerate(data["streams"]):
                augmented_stream = data["augmented_streams"][stream_index]

                all_removed_events_from_stream_index = [event for event in removed_events if event[0][0] == stream_index]
                while len(all_removed_events_from_stream_index) > 0:
                    last_event = all_removed_events_from_stream_index.pop()
                    from_index = last_event[-1][1] + 1
                    keep_until_index = last_event[0][1]

                    # Keep the stream from the end of the event onward
                    new_streams.append(stream[from_index:])
                    new_augmented_streams.append(augmented_stream[from_index:])

                    # Truncate the stream from the start of the deleted event onward
                    stream = stream[:keep_until_index]
                    augmented_stream = augmented_stream[:keep_until_index]

            data["streams"] = new_streams
            data["augmented_streams"] = new_augmented_streams

    for stream_index, stream in enumerate(data["streams"]):
        total_stream = []
        for stream_frame_index, stream_frame in enumerate(stream):
            total_stream.append([stream_index, stream_frame_index, stream_frame[1]])

            if stream_frame_index > sequence_length - 1:
                last_sequence_length_frames = total_stream[-sequence_length:]

                if stream_frame[1] == label:
                    data["label"].append(last_sequence_length_frames)
                    data["augmented"].append(last_sequence_length_frames)

                elif stream_frame[1] == BACKGROUND_LABEL:
                    data["background"].append(last_sequence_length_frames)
                    data["background_augmented"].append(last_sequence_length_frames)

    return data
