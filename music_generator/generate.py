from argparse import ArgumentParser
from copy import deepcopy
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import datetime
from utils import get_n_instruments
from models.build_model import build_model
from data.data_processing_reverse import ind_tensor_to_mid, ind_tensor_to_str
import pandas as pd
from tqdm import tqdm

from create_dataset.utils import get_target_seq


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate(model, maps, device, out_dir, conditioning, short_filename=False,
                penalty_coeff=0.5, continuous_conditions=None,
                    max_input_len=1024, amp=True, step=None, 
                    gen_len=2048, temperatures=[1.2,1.2], top_k=-1, 
                    top_p=0.7, debug=False, varying_condition=None, seed=-1,
                    verbose=False, primers=[["<START>"]], min_n_instruments=2, sceneid=0, demo=False):

    if not debug:
        os.makedirs(out_dir, exist_ok=True)
    
    model = model.to(device)
    model.eval()

    assert len(temperatures) in (1, 2)

    try:
        # changed
        continuous_conditions = continuous_conditions.float().to(device)
    except:
        continuous_conditions = None
    if conditioning == "none":
        batch_size = len(primers)
    elif conditioning == "continuous_concat":
        batch_size = len(continuous_conditions)
            
    # will be used to penalize repeats
    repeat_counts = [0 for _ in range(batch_size)]

    exclude_symbols = [symbol for symbol in maps["tuple2idx"].keys() if symbol[0] == "<"]

    # will have generated symbols and indices
    gen_song_tensor = torch.LongTensor([]).to(device)

    if not isinstance(primers, list):
        primers = [[primers]]
    primer_inds = [[maps["tuple2idx"][symbol] for symbol in primer] \
        for primer in primers]

    gen_inds = torch.LongTensor(primer_inds)

    null_conditions_tensor = torch.FloatTensor([np.nan, np.nan]).to(device)

    if len(primers) == 1:
        gen_inds = gen_inds.repeat(batch_size, 1)
        null_conditions_tensor = null_conditions_tensor.repeat(batch_size, 1)

    if conditioning == "continuous_concat":
        conditions_tensor = continuous_conditions
        
        # changed : shape of condition : (B, L, 2) -> (B, gen_len, 2)
        B, L, _ = conditions_tensor.shape
        if L >= gen_len:
            conditions_tensor = conditions_tensor[:, :gen_len]
        else:
            # When L=4, gen_len=10, [1, 2, 3, 4] -> [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
            # rem elements at front are repeated quo+1 times and other elements are repeated quo times
            quo = gen_len // L
            rem = gen_len % L
            new_condition = [] # list of tensor (B, 2)
            for i in range(L):
                if i <= rem - 1:
                    new_condition.extend([conditions_tensor[:, i]] * (quo + 1))
                else:
                    new_condition.extend([conditions_tensor[:, i]] * quo)
            conditions_tensor = torch.stack(new_condition, dim=1) # shape (B, gen_len, 2)
    else:
        conditions_tensor = null_conditions_tensor


    gen_inds = gen_inds.t().to(device)

    with torch.no_grad():
        i = 0
        while i < gen_len:
            i += 1
            if verbose:
                print(gen_len - i, end=" ", flush=True)

            # changed
            gen_song_tensor = torch.cat((gen_song_tensor, gen_inds), 0)
            
            input_ = gen_song_tensor
            if len(gen_song_tensor) > max_input_len:
                input_ = input_[-max_input_len:, :]

            # changed
            if conditioning == "continuous_concat":
                conditions_tensor_cut = conditions_tensor[:, :i]
            else:
                conditions_tensor_cut = conditions_tensor

            # Run model
            with torch.cuda.amp.autocast(enabled=amp):
                input_ = input_.t()
                output = model(input_, conditions_tensor_cut) # input_ : shape (i, 2) / conditions_tensor : shape (B, i, 2)
                output = output.permute((1, 0, 2)) # shape (self.input_len, B, score)

            # Process output, get predicted token
            output = output[-1, :, :]     # Select last timestep # shape (B, score)
            output[output != output] = 0    # zeroing nans
            
            if torch.all(output == 0) and verbose:
                # if everything becomes zero
                print("All predictions were NaN during generation")
                output = torch.ones(output.shape).to(device)

            # exclude certain symbols
            for symbol_exclude in exclude_symbols:
                try:
                    idx_exclude = maps["tuple2idx"][symbol_exclude]
                    output[:, idx_exclude] = -float("inf")
                except:
                    pass

            effective_temps = []
            for j in range(batch_size):
                gen_idx = gen_inds[0, j].item()
                gen_tuple = maps["idx2tuple"][gen_idx]

                # changed
                if conditioning == "continuous_concat":
                    effective_temp = round((-0.15) * (float(torch.mean(continuous_conditions[j], dim=0).tolist()[1]) + 1)**2 + 1.8, 1)  # shape (2,)
                else:
                    effective_temp = temperatures[1]
                
                if isinstance(gen_tuple, tuple):
                    gen_event = maps["idx2event"][gen_tuple[0]]
                    if "TIMESHIFT" in gen_event:
                        if conditioning == "continuous_concat":
                            effective_temp = round((-0.15) * (float(torch.mean(continuous_conditions[j], dim=0).tolist()[1]) + 1)**2 + 1.8, 1)  # shape (2,)
                        else:
                            effective_temp = temperatures[0]
                effective_temps.append(effective_temp)

            temp_tensor = torch.Tensor([effective_temps]).to(device)

            output = F.log_softmax(output, dim=-1) # shape (B, score)
            
            # Add repeat penalty to temperature
            if penalty_coeff > 0:
                repeat_counts_array = torch.Tensor(repeat_counts).to(device)
                temp_multiplier = torch.maximum(torch.zeros_like(repeat_counts_array, device=device), 
                    torch.log((repeat_counts_array+1)/4)*penalty_coeff)
                repeat_penalties = temp_multiplier * temp_tensor
                temp_tensor += repeat_penalties

            # Apply temperature
            output /= temp_tensor.t()
            
            # top-k
            if top_k <= 0 or top_k > output.size(-1): 
                top_k_eff = output.size(-1)
            else:
                top_k_eff = top_k
            output, top_inds = torch.topk(output, top_k_eff) # shape (B, top_k_score)

            # top-p
            if top_p > 0 and top_p < 1:
                cumulative_probs = torch.cumsum(F.softmax(output, dim=-1), dim=-1)
                remove_inds = cumulative_probs > top_p
                remove_inds[:, 0] = False   # at least keep top value
                output[remove_inds] = -float("inf")

            output = F.softmax(output, dim=-1)
        
            # Sample from probabilities
            inds_sampled = torch.multinomial(output, 1, replacement=True)
            gen_inds = top_inds.gather(1, inds_sampled).t()

            # Update repeat counts
            num_choices = torch.sum((output > 0).int(), -1)
            for j in range(batch_size):
                if num_choices[j] <= 2: repeat_counts[j] += 1
                else: repeat_counts[j] = repeat_counts[j] // 2

        # Convert to midi and save

        # If there are less than n instruments, repeat generation for specific condition
        redo_primers, redo_continuous_conditions = [], []
        for i in range(gen_song_tensor.size(-1)):
            if short_filename:
                if demo:
                  out_file_path = f"_{sceneid}"
                else:
                  out_file_path = f"_{i}"
            else:
                if step is None:
                    now = datetime.datetime.now()
                    out_file_path = now.strftime("%Y_%m_%d_%H_%M_%S")
                else:
                    out_file_path = step

                if demo:
                  out_file_path += f"_{sceneid}"
                else:
                  out_file_path += f"_{i}"

            if seed > 0:
                out_file_path += f"_s{seed}"

            if continuous_conditions is not None:
                condition = torch.mean(continuous_conditions[i], dim=0).tolist() # shape (2,)
                # convert to string
                condition = [str(round(c, 2)).replace(".", "") for c in condition]
                out_file_path += f"_V{condition[0]}_A{condition[1]}"

            out_file_path += ".mid"
            out_path_mid = os.path.join(out_dir, out_file_path)

            symbols = ind_tensor_to_str(gen_song_tensor[:, i], maps["idx2tuple"], maps["idx2event"])
            n_instruments = get_n_instruments(symbols)

            if verbose:
                print("")
            if n_instruments >= min_n_instruments:
                mid = ind_tensor_to_mid(gen_song_tensor[:, i], maps["idx2tuple"], maps["idx2event"], verbose=False)
                out_path_txt = "txt_" + out_file_path.replace(".mid", ".txt")
                out_path_txt = os.path.join(out_dir, out_path_txt)
                out_path_inds = "inds_" + out_file_path.replace(".mid", ".pt")
                out_path_inds = os.path.join(out_dir, out_path_inds)

                if not debug:
                    mid.write(out_path_mid)
                    if verbose:
                        print(f"Saved to {out_path_mid}")
            else:
                print(f"Only has {n_instruments} instruments, not saving.")
                if conditioning == "none":
                    redo_primers.append(primers[i])
                    redo_continuous_conditions = None
                else:
                    redo_continuous_conditions.append(continuous_conditions[i, :].tolist())
                    redo_primers = primers

    return redo_primers, redo_continuous_conditions


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_model_dir = os.path.abspath(os.path.join(script_dir, 'model'))
    code_utils_dir = os.path.join(code_model_dir, 'utils')
    sys.path.extend([code_model_dir, code_utils_dir])

    parser = ArgumentParser()

    parser.add_argument('--model_dir', type=str, help='Directory with model', required=True)
    parser.add_argument('--no_cuda', action='store_true', help="Use CPU")
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    parser.add_argument('--gen_len', type=int, help='Max generation len', default=512)
    parser.add_argument('--max_input_len', type=int, help='Max input len', default=5000) # original : 1216
    parser.add_argument('--temp', type=float, nargs='+', help='Generation temperature', default=[1.2, 1.2])
    parser.add_argument('--topk', type=int, help='Top-k sampling', default=-1)
    parser.add_argument('--topp', type=float, help='Top-p sampling', default=0.7)
    parser.add_argument('--debug', action='store_true', help="Do not save anything")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--no_amp', action='store_true', help="Disable automatic mixed precision")
    parser.add_argument("--conditioning", type=str, required=True,
                    choices=["none", "discrete_token", "continuous_token",
                             "continuous_concat"], help='Conditioning type')
    parser.add_argument('--penalty_coeff', type=float, default=0.5,
                        help="Coefficient for penalizing repeating notes")
    parser.add_argument("--quiet", action='store_true', help="Not verbose")
    parser.add_argument("--short_filename", action='store_true')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--min_n_instruments', type=int, help='Minimum number of instruments', default=1)
    parser.add_argument("--batch_gen_dir", type=str, default="")

    args = parser.parse_args()

    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    main_output_dir = "./output"
    assert os.path.exists(os.path.join(main_output_dir, args.model_dir))
    midi_output_dir = os.path.join(main_output_dir, args.model_dir, "generations", "inference")

    new_dir = ""
    if args.batch_gen_dir != "":
        new_dir = new_dir + "_" + args.batch_gen_dir
    if new_dir != "":
        midi_output_dir = os.path.join(midi_output_dir, new_dir)
    if not args.debug:
        os.makedirs(midi_output_dir, exist_ok=True)

    model_fp = os.path.join(main_output_dir, args.model_dir, 'model.pt')
    mappings_fp = os.path.join(main_output_dir, args.model_dir, 'mappings.pt')
    config_fp = os.path.join(main_output_dir, args.model_dir, 'model_config.pt')

    if os.path.exists(mappings_fp):
        maps = torch.load(mappings_fp)
    else:
        raise ValueError("Mapping file not found.")

    start_symbol = "<START>"
    n_emotion_bins = 5
    valence_symbols, arousal_symbols = [], []

    emotion_bins = np.linspace(-1-1e-12, 1+1e-12, num=n_emotion_bins+1)
    if n_emotion_bins % 2 == 0:
        bin_ids = list(range(-n_emotion_bins//2, 0)) + list(range(1, n_emotion_bins//2+1))
    else:
        bin_ids = list(range(-(n_emotion_bins-1)//2, (n_emotion_bins-1)//2 + 1))
        
    for bin_id in bin_ids:
        valence_symbols.append(f"<V{bin_id}>")
        arousal_symbols.append(f"<A{bin_id}>")

    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    
    verbose = not args.quiet
    if verbose:
        if device == torch.device("cuda"):
            print("Using GPU")
        else:
            print("Using CPU")

    # Load model
    config = torch.load(config_fp)
    model, _ = build_model(None, load_config_dict=config)
    model = model.to(device)
    if os.path.exists(model_fp):
        model.load_state_dict(torch.load(model_fp, map_location=device))
    elif os.path.exists(model_fp.replace("best_", "")):
        model.load_state_dict(torch.load(model_fp.replace("best_", ""), map_location=device))
    else:
        raise ValueError("Model not found")

    primers = [["<START>"]]
    
    if args.conditioning == "none":
        primers = [["<START>"] for _ in range(args.batch_size)]

    elif args.conditioning == "continuous_concat":
        primers = [["<START>"]]

    # demo_path = "../emotion_recognizer/output/va_scores.csv"
    demo_path = "../emotion_recognizer/output/scene_va.csv"
    # demo_path = "../emotion_recognizer/output/va_scores_5th_model_10epoch.csv"
    # demo_path = "../emotion_recognizer/output/va_scores_5th_model_10epoch(2).csv"
    demo_df = pd.read_csv(demo_path)
    scene_ids = demo_df["scene_id"]


    # scene_id : batch
    for scene_id in scene_ids:
        # extract va array of scene_id
        va = demo_df[demo_df["scene_id"] == scene_id]
        valence = [float(x) for x in str(va["valence"].tolist()).replace("['", "").replace("']", "").replace("'[", "").replace("]'", "").strip().replace("[", "").replace("]", "").split()]
        arousal = [float(x) for x in str(va["arousal"].tolist()).replace("['", "").replace("']", "").replace("'[", "").replace("]'", "").strip().replace("[", "").replace("]", "").split()]

        
        conditions = []
        for i in range(len(valence)):
            conditions.append([valence[i], arousal[i]])
        continuous_conditions = torch.tensor(conditions).unsqueeze(0).to(device) # shape (1, L, 2)
          
        for i in range(args.num_runs):
            primers_run = deepcopy(primers)
            continuous_conditions_run = deepcopy(continuous_conditions)
            while not (primers_run == [] or continuous_conditions_run == []):
                primers_run, continuous_conditions_run = generate(
                            model, maps, device,
                            midi_output_dir, args.conditioning,
                            min_n_instruments=args.min_n_instruments,continuous_conditions=continuous_conditions_run,
                            penalty_coeff=args.penalty_coeff, short_filename=args.short_filename, top_p=args.topp,
                            gen_len=args.gen_len, max_input_len=args.max_input_len,
                            amp=not args.no_amp, primers=primers_run, temperatures=args.temp, top_k=args.topk,
                            debug=args.debug, verbose=not args.quiet, seed=args.seed, sceneid = scene_id, demo = True)
