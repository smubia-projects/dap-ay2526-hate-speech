import json
import random
from collections import defaultdict
import os

def stratified_split(filepath, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    random.seed(seed)
    
    # Load all entries
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    # Group by label for stratification
    label_groups = defaultdict(list)
    for entry in data:
        label_groups[entry['label']].append(entry)
    
    train, val, test = [], [], []
    
    for label, group in label_groups.items():
        random.shuffle(group)
        n = len(group)
        # Calculate split indices
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train+n_val])
        test.extend(group[n_train+n_val:])
        
    return train, val, test

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    input_file = "/Users/shariff/Documents/github/11th-DAP-hate-speech/data/singlish/synthetic/synthetic_singlish_20_terms_consolidated.jsonl"
    repaired_dir = "/Users/shariff/Documents/github/11th-DAP-hate-speech/data/singlish/synthetic/20-terms_refined/repaired_splits_60_20_20/"
    os.makedirs(repaired_dir, exist_ok=True)
    
    train, val, test = stratified_split(input_file)
    
    save_jsonl(train, os.path.join(repaired_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(repaired_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(repaired_dir, "test.jsonl"))
    
    print(f"Total entries: {len(train) + len(val) + len(test)}")
    print(f"Train (60%): {len(train)}")
    print(f"Val (20%):   {len(val)}")
    print(f"Test (20%):  {len(test)}")
    print(f"Splits saved in: {repaired_dir}")
