import csv
import json
import pretty_midi
from tqdm import tqdm
import os
import utils
import pandas as pd
from copy import deepcopy

"""
Creates labels for DEAM dataset.
Labels include low-level MIDI features such as tempo, note density and number of MIDI files.
They also include high-level features, such as valence, arousal, etc.
"""
write = True
redo = True

output_dir = "../../../data/DEAM/features"
os.makedirs(output_dir, exist_ok=True)

valence_path = "../../../data/DEAM/valence.csv"
arousal_path = "../../../data/DEAM/arousal.csv"
midi_dataset_path = "../../../data/DEAM/midi_files/"
extension = ".mid"


### PART I: Map song_ids (in midi dataset) to VA features

### 1- Merge and add VA features
output_path_incomplete = os.path.join(output_dir, "incomplete_songid_to_va_features.csv")
output_path = os.path.join(output_dir, "songid_to_va_features.json")

if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        songid_to_va_features = json.load(f)
else:
    fieldnames = ["song_id", "valence", "arousal"]
    with open(output_path_incomplete, "w") as f_out:
        csv_writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        csv_writer.writeheader()
        midi_paths = utils.get_dir_files(midi_dataset_path, extension)
        song_ids = utils.get_song_ids(midi_paths)
        valence_df = pd.read_csv(valence_path)
        arousal_df = pd.read_csv(arousal_path)

        print("Adding VA features")
        data = {}
        for song_id in tqdm(song_ids):
            data["song_id"] = song_id
            data["valence"] = utils.get_target_seq(valence_df, song_id)
            data["arousal"] = utils.get_target_seq(arousal_df, song_id)
            csv_writer.writerow(data)

    # Now write final data to json
    songid_to_va_features_list = utils.read_csv(output_path_incomplete)
    songid_to_va_features = {}
    # unlike json, csv doesnt support dict within dict, so convert it to dict manually
    for item in songid_to_va_features_list:
        song_id = deepcopy(item["song_id"])
        del item["song_id"]
        songid_to_va_features[song_id] = item

    if write:   
        with open(output_path, "w") as f:
            json.dump(songid_to_va_features, f, indent=4)
            print(f"Output saved to {output_path}")

### PART II: Dealing with symbolic music data

### 2- Filter unique midis
# 일단 남겨는 뒀지만, 추후 상황을 보며 deprecate하거나 삭제 가능
"""Dataset was created by creating hashes for the entire files
and then keeping files with unique hashes.
However, some files' musical content are the same, and only their metadata are different.
So we hash the content (pianoroll array), and further filter out the unique ones."""

# Create hashes for midis
output_path = os.path.join(output_dir, "hashes.json")

if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_file_to_hash = json.load(f)
else:
    def get_hash_and_file(path):
        hash_ = utils.get_hash(midi_dataset_path + path)
        file_ = os.path.basename(path)
        fname = file_.split('.')    # leave only song_id
        return [int(fname[0]), hash_]

    file_paths = sorted(utils.get_dir_files(midi_dataset_path, extension))
    assert len(file_paths) > 0, f"No MIDI files found"
    print("Getting hashes for MIDIs.")
    midi_file_to_hash = []
    for file_path in tqdm(file_paths):
        midi_file_to_hash.append(get_hash_and_file(file_path))
    midi_file_to_hash = sorted(midi_file_to_hash, key=lambda x:x[0])
    midi_file_to_hash = dict(midi_file_to_hash)
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_file_to_hash, f, indent=4)
            print(f"Output saved to {output_path}")

# also do the reverse hash -> midi
output_path = os.path.join(output_dir, "unique_files.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_files_unique = json.load(f)
else:
    hash_to_midi_file = {}
    for midi_file, hash in midi_file_to_hash.items():
        if hash in hash_to_midi_file.keys():
            hash_to_midi_file[hash].append(midi_file)
        else:
            hash_to_midi_file[hash] = [midi_file]

    midi_files_unique = []
    # Get unique midis (with highest match score)
    print("Getting unique MIDIs.")
    for hash, midi_files in tqdm(hash_to_midi_file.items()):
        if hash != "empty_pianoroll":
            midi_files_unique.append(midi_files[0])
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_files_unique, f, indent=4)
            print(f"Output saved to {output_path}")

### 3- For all midis, get low level features 
# (tempo, note density, number of instruments)

output_path = os.path.join(output_dir, "midi_features.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_file_to_midi_features = json.load(f)
else:
    def get_midi_features(midi_file):
        mid = pretty_midi.PrettyMIDI(midi_dataset_path + midi_file + '.mid')
        note_density = utils.get_note_density(mid)
        tempo = utils.get_tempo(mid)
        n_instruments = utils.get_n_instruments(mid)
        duration = mid.get_end_time()
        midi_features = {
            "note_density": note_density,
            "tempo": tempo,
            "n_instruments": n_instruments,
            "duration": duration,
        }
        return [midi_file, midi_features]

    print("Getting low-level MIDI features")
    midi_file_to_midi_features = []
    for midi_file_unique in tqdm(midi_files_unique):
        midi_file_to_midi_features.append(get_midi_features(str(midi_file_unique)))
    midi_file_to_midi_features = dict(midi_file_to_midi_features)
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_file_to_midi_features, f, indent=4)
            print(f"Output saved to {output_path}")

### 4- Merge MIDI features and matched VA features
output_path = os.path.join(output_dir, "full_dataset_features.json")
if os.path.exists(output_path) and not redo: 
    with open(output_path, "r") as f:
        midi_file_to_merged_features = json.load(f)
else:
    midi_file_to_merged_features = {}
    print("Merging MIDI features and VA features for full dataset.")
    for midi_file in tqdm(midi_file_to_midi_features.keys()):
        midi_file_to_merged_features[midi_file] = {}
        midi_file_to_merged_features[midi_file]["midi_features"] = midi_file_to_midi_features[midi_file]
        if midi_file in songid_to_va_features.keys():
            matched_features = songid_to_va_features[midi_file]
        else:
            matched_features = {}
        midi_file_to_merged_features[midi_file]["matched_features"] = matched_features
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_file_to_merged_features, f, indent=4)
            print(f"Output saved to {output_path}")

### PART III: Constructing training dataset
### 5- Summarize matched dataset features by only taking valence and note densities per instrument,
# number of instruments, durations, is_matched

output_path = os.path.join(output_dir, "full_dataset_features_summarized.csv")
if not os.path.exists(output_path) or redo:
    print("Constructing training dataset (final file)")
    dataset_summarized = []
    for midi_file, features in tqdm(midi_file_to_merged_features.items()):
        midi_features = features["midi_features"]
        n_instruments = midi_features["n_instruments"]
        note_density_per_instrument = midi_features["note_density"] / n_instruments
        matched_features = features["matched_features"]
        if matched_features == {}:
            is_matched = False
            valence = float("nan")
            arousal = float("nan")
        else:
            is_matched = True
            valence_features = matched_features["valence"]
            arousal_features = matched_features["arousal"]
            if valence_features is None or len(valence_features) < 1:
                valence = float("nan")
            else:
                valence = valence_features
            if arousal_features is None or len(arousal_features) < 1:
                arousal = float("nan")
            else:
                arousal = arousal_features
        
        dataset_summarized.append({
            "file": midi_file,
            "is_matched": is_matched,
            "n_instruments": n_instruments,
            "note_density_per_instrument": note_density_per_instrument,
            "valence": valence,
            "arousal": arousal
        })
    dataset_summarized = pd.DataFrame(dataset_summarized)
    if write:
        dataset_summarized.to_csv(output_path, index=False)
        print(f"Output saved to {output_path}")