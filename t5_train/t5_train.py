import argparse
import json
import os

import nltk
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from dataset import BashDataset
import warnings

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model_params = {}


def count_tr_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def print_model_parameters(model: torch.nn.Module):
    total_params = count_all_parameters(model)
    trainable_params = count_tr_parameters(model)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    g_loss = 0.
    valid_steps = 0
    for _, data in enumerate(tqdm(loader, desc=f"Train [ {epoch} / {model_params['n_epochs']} ]", leave=False), 0):

        y = data["target_ids"].to(device, dtype=torch.long)
        lm_labels = y.clone()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=lm_labels,
        )
        loss = outputs[0].mean()  # multiple losses from data parallel
        if torch.isnan(loss).any():
            print("nan in loss")
            continue
        valid_steps += 1
        g_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    return float(g_loss) / max(1, valid_steps)


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    g_loss = 0.0
    valid_steps = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, desc=f"Valid [ {epoch} / {model_params['n_epochs']} ]", leave=False)):
            # prepare inputs
            ids = batch["source_ids"].to(device)
            mask = batch["source_mask"].to(device)
            y = batch["target_ids"].to(device)
            lm_labels = y.clone()
            lm_labels[lm_labels == tokenizer.pad_token_id] = -100

            # forward pass for loss
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=lm_labels,
            )
            loss = outputs[0].mean()  # multiple losses from data parallel
            if torch.isnan(loss).any():
                print("nan in loss")
                continue
            valid_steps += 1
            g_loss += loss.item()

    return float(g_loss) / max(1, valid_steps)


def load_data_dir(d_path: str) -> list:
    with open(os.path.join(d_path, "shio_data.ndjson"), 'r') as f:
        dataset_records = [json.loads(line) for line in f]
    return dataset_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False, default="t5_train/t5_config.json", help="path to run config")

    return parser.parse_args()


def main():
    args = parse_args()
    global model_params
    with open(args.config, 'r') as f:
        model_params = json.load(f)

    torch.manual_seed(model_params["seed"])
    np.random.seed(model_params["seed"])

    nltk.download('punkt', quiet=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")  # codet5
    print("Tokenizer loaded")

    datasets = []


    for datadir in os.listdir(model_params["data_path"]):
        data = load_data_dir(os.path.join(model_params["data_path"], datadir))
        tr_data, e_data = train_test_split(data, test_size=0.2, random_state=model_params['seed'])  # 80-10-10
        vl_data, ts_data = train_test_split(e_data, test_size=0.5, random_state=model_params['seed'])
        datasets.append([datadir, tr_data, vl_data])
        print(f"{datadir}: {len(data)} samples")

    os.makedirs(model_params['model_path'], exist_ok=True)

    for data_name, tr_data, vl_data in datasets:
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(device)

        if model_params["multi_gpu"]:
            model = torch.nn.DataParallel(model)

        tr_dataset = BashDataset(tr_data, tokenizer=tokenizer, i_len=model_params['max_i_len'], o_len=model_params['max_o_len'])
        tr_loader = DataLoader(tr_dataset, batch_size=model_params['b_size'], shuffle=True)
        vl_dataset = BashDataset(vl_data, tokenizer=tokenizer, i_len=model_params['max_i_len'], o_len=model_params['max_o_len'])
        vl_loader = DataLoader(vl_dataset, batch_size=model_params['b_size'], shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=model_params['lr'], weight_decay=0.01)

        for epoch in range(model_params['n_epochs']):
            t_l = train(epoch+1, tokenizer, model, device, tr_loader, optimizer)
            v_l = validate(epoch+1, tokenizer, model, device, vl_loader)
            header = f"[ {data_name} ]"
            print(f"{header:<20}[ {epoch + 1} / {model_params['n_epochs']} ] Train Loss: {t_l:.4f} | Val Loss: {v_l:.4f}")

        model_path = os.path.join(model_params['model_path'], data_name)
        os.makedirs(model_path, exist_ok=True)  # Ensure the directory exists
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        model_to_save.save_pretrained(model_path)

        del model, optimizer, tr_loader, vl_loader  # free GPU memory
        torch.cuda.empty_cache()
        print(f"Weights saved to {model_path}")


if __name__ == "__main__":
    main()
