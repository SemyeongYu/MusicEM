import pandas as pd
import numpy as np
import csv

def preprocess_features(feature_file, min_n_instruments=3,
        test_ratio=0.05, outlier_range=1.5, conditional=True,
        use_labeled_only=True):

    # Preprocess data
    data = pd.read_csv(feature_file)
    mapper = {"valence": "valence", "arousal": "arousal"}
    data = data.rename(columns=mapper)
    columns = data.columns.to_list()

    # filter out ones with less instruments
    data = data[data["n_instruments"] >= min_n_instruments]
    # changed
    data = data[data["file"] != 2011]
    data = data[data["file"] != 2026]
    
    '''
    # changed
    fields = ['file']
    rows = [[song_id] for song_id in data["file"]]
    with open('used_song', 'w') as f:
      write = csv.writer(f)
      write.writerow(fields)
      write.writerows(rows) 
    '''
    feature_labels = list(mapper.values())

    # convert NaN into None
    data = data.where(pd.notnull(data), None)

    # Create train and test splits
    matched = data[data["is_matched"]]
    unmatched = data[~data["is_matched"]]

    # reserve a portion of matched data for testing
    matched = matched.sort_values("file")
    matched = matched.reset_index(drop=True)
    n_test_samples = round(len(matched) * test_ratio)

    test_split = matched.loc[len(matched)-n_test_samples:len(matched)]

    train_split = matched.loc[:len(matched)-n_test_samples]

    if not use_labeled_only:
        train_split = pd.concat([train_split, unmatched])
        train_split = train_split.sort_values("file").reset_index(drop=True)

    splits = [train_split, test_split]

    # summarize
    columns_to_drop = [col for col in columns if col not in ["file", "valence", "arousal"]]
    if not conditional:
        columns_to_drop += ["valence", "arousal"]

    # filter data so all features are valid (not None = matched data)
    for label in feature_labels:
        # test split has to be identical across vanilla and conditional models
        splits[1] = splits[1][~splits[1][label].isnull()]

        # filter train split only for conditional models
        if use_labeled_only:
            splits[0] = splits[0][~splits[0][label].isnull()]

    for i in range(len(splits)):
        # summarize
        splits[i] = splits[i].drop(columns=columns_to_drop, errors="ignore")
        splits[i] = splits[i].to_dict("records")    

    return splits