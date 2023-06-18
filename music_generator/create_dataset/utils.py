import re
import hashlib
import pypianoroll
import numpy as np
import pandas as pd
import pretty_midi
import csv
import os
from glob import glob

def get_hash(path):
    if path[-4:] == ".mid":
        try:
            mid = pretty_midi.PrettyMIDI(path)
        except:
            return "empty_pianoroll"
        try:
            rolls = mid.get_piano_roll()
        except:
            return "empty_pianoroll"
        if rolls.size == 0:
            return "empty_pianoroll"
    else:
        pr = pypianoroll.load(path)
        tracks = sorted(pr.tracks, key=lambda x: x.name)
        rolls = [track.pianoroll for track in tracks if track.pianoroll.shape[0] > 0]
        if rolls == []:
            return "empty_pianoroll"
        rolls = np.concatenate(rolls, axis=-1)
    hash_ = hashlib.sha1(np.ascontiguousarray(rolls)).hexdigest()
    return hash_

def get_note_density(mid):
    duration = mid.get_end_time()
    n_notes = sum([1 for instrument in mid.instruments for note in instrument.notes])
    density = n_notes / duration
    return density

def get_tempo(mid):
    tick_scale = mid._tick_scales[-1][-1]
    resolution = mid.resolution
    beat_duration = tick_scale * resolution
    mid_tempo = 60 / beat_duration
    return mid_tempo

def get_n_instruments(mid):
    n_instruments = sum([1 for instrument in mid.instruments if instrument.notes != []])
    return n_instruments

def fix_string(s):
    if s != "":
        s = s.lower()   # lowercase
        s = s.replace('\'s', '')    # remove 's
        s = s.replace('_', ' ')    # remove _
        s = re.sub("[\(\[].*?[\)\]]", "", s)    # remove everything in parantheses
        if s[-1] == " ":    # remove space at the end
            s = s[:-1]
    return s

def read_csv(input_file_path, delimiter=","):
    with open(input_file_path, "r") as f_in:
        reader = csv.DictReader(f_in, delimiter=delimiter)
        data = [{key: value for key, value in row.items()} for row in reader]
    return data

def extract_song_id(filepath):
    res = filepath.split('/')[-1]   # res[-1] == sog_id.mp3
    return int(res.split('.')[0])    # extension 제거

def get_song_ids(files):
    song_ids = [extract_song_id(f) for f in files]
    return song_ids

def get_dir_files(dir_path, extension):
    filenames = os.listdir(os.path.abspath(dir_path))
    file_mid = [file for file in filenames if file.endswith(extension)]
    return file_mid

def get_target_seq(df: pd.DataFrame, song_id):
    df = df.loc[df['song_id'] == song_id]
    df = df.drop(['song_id'], axis=1)
    df = df.dropna(axis=1)
    return df.to_numpy()[0]