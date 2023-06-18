import json
from data_processing import read_pianoroll, mid_to_bars, get_maps
import torch
# import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time
from functools import partial
import os
import pretty_midi

""" Preprocessing Lakh MIDI pianoroll dataset.
Divides into bars. Encodes into tuples. Makes transposing easier. """

def run(f, my_iter):
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    return results

def process(pr_path, event_sym2idx):
    time.sleep(0.001)

    mid = pretty_midi.PrettyMIDI(pr_path)
    bars = mid_to_bars(mid, event_sym2idx)

    # file_ = pr_path.split("/")[-1]
    file_ = os.path.basename(pr_path)
    file_ = int(file_.split('.')[0])
    
    item_data = {
                "file": file_,
                "bars": bars, 
                 }

    return item_data

def main():

    main_dir = "../../data_files/data"
    input_dir = "../../data_files/midi_files"
    unique_pr_list_file = "../../data_files/features/unique_files.json"

    output_dir = os.path.join(main_dir, "deam_transposable")

    os.makedirs(output_dir, exist_ok=True)
    output_maps_path = os.path.join(main_dir, "maps.pt")

    with open(unique_pr_list_file, "r") as f:
        pr_paths = json.load(f)

    # deam dataset과 lakh dataset의 디렉터리 구조 차이 때문에 pr_path[0]을 디렉토리로 추가하는게 의미없어서 제거함
    pr_paths = [os.path.join(input_dir, str(pr_path) + ".mid") for pr_path in pr_paths]

    maps = get_maps()
    
    func = partial(process, event_sym2idx=maps["event2idx"])

    os.makedirs(output_dir, exist_ok=True)

    x = run(func, pr_paths)
    x = [item for item in x if item["bars"] is not None]
    for i in tqdm(range(len(x))):
        for j in range(len(x[i]["bars"])):
            x[i]["bars"][j] = torch.from_numpy(x[i]["bars"][j])
        fname = x[i]["file"]
        output_path = os.path.join(output_dir, str(fname) + ".pt")
        torch.save(x[i], output_path)

    torch.save(maps, output_maps_path)
    

if __name__ == "__main__":
    main()





