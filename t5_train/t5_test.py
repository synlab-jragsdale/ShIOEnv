import argparse
import json
import os

import nltk
import numpy as np
import torch
from sacrebleu import CHRF
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataset import BashDataset
import warnings

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector")

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

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


def compute_bleu(epoch, tokenizer, model, device, loader, max_length=512, num_beams=4):
    """
    Runs the model in generation mode over loader,
    computes a smoothed sentence BLEU for each example
    against its single reference, and returns the average.
    """
    model.eval()
    smoothie = SmoothingFunction().method4
    total_bleu = 0.0
    n_examples = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, desc=f"BLEU [ {epoch} / {model_params['n_epochs']} ]", leave=False)):
            # move to device
            input_ids = batch["source_ids"].to(device)
            attention_mask = batch["source_mask"].to(device)

            # generate with T5
            generated_ids = model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True if num_beams > 1 else False
            )

            # decode predictions + references
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(batch["target_ids"], skip_special_tokens=True)

            # compute BLEU per example
            for pred, ref in zip(preds, refs):
                # tokenize on whitespace
                pred_tokens = pred.split()
                ref_tokens = ref.split()

                # if empty pred or ref, bleu=0
                if len(pred_tokens)==0 or len(ref_tokens)==0:
                    bleu_score = 0.0
                else:
                    bleu_score = sentence_bleu(
                        [ref_tokens],
                        pred_tokens,
                        smoothing_function=smoothie
                    )

                total_bleu += bleu_score
                n_examples += 1

    return total_bleu / max(1, n_examples)


def compute_chrf(epoch, tokenizer, model, device, loader, max_length=512, num_beams=4):
    """
    Runs the model in generation mode,
    computes corpus‑level chrF (character F-score) on the full set
    """
    model.eval()
    metric = CHRF(word_order=2)  # chrF (default) word_order=0-> chrF++
    preds, refs = [], []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, desc=f"chrF [ {epoch} / {model_params['n_epochs']} ]", leave=False)):
            # move to device
            inp_ids = batch["source_ids"].to(device)
            attn_m = batch["source_mask"].to(device)

            # generate
            gen_ids = model.module.generate(
                input_ids=inp_ids,
                attention_mask=attn_m,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

            # decode
            preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            refs.extend(tokenizer.batch_decode(batch["target_ids"], skip_special_tokens=True))

    # corpus‑level chrF (more stable than sentence avg.)
    score = metric.corpus_score(preds, [refs]).score
    return score


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

    ts_data = []  # dataset
    ts_inputs = []  # keep track of all used inputs, no duplication in test set

    for datadir in os.listdir(model_params["data_path"]):
        data = load_data_dir(os.path.join(model_params["data_path"], datadir))
        _, _data = train_test_split(data, test_size=0.2, random_state=model_params['seed'])  # 80-10-10
        _, all_test_data = train_test_split(_data, test_size=0.5, random_state=model_params['seed'])
        for sample in all_test_data:
            # if nonredundant sample and not already in testing set
            if sample["redundancy_reward"] > 0.9 and sample["input"] not in ts_inputs:
                ts_data.append(sample)
                ts_inputs.append(sample["input"])

    for datadir in os.listdir(model_params["data_path"]):
        model_path = str(os.path.join(model_params['model_path'], datadir))
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        if model_params["multi_gpu"]:
            model = torch.nn.DataParallel(model)

        ts_dataset = BashDataset(ts_data, tokenizer=tokenizer, i_len=model_params['max_i_len'], o_len=model_params['max_o_len'])
        ts_loader = DataLoader(ts_dataset, batch_size=model_params['b_size'], shuffle=False)
        t_l = validate(0, tokenizer, model, device, ts_loader)
        t_bleu = compute_bleu(model_params['n_epochs'], tokenizer, model, device, ts_loader, max_length=model_params['max_o_len'])
        t_chrF = compute_chrf(model_params['n_epochs'], tokenizer, model, device, ts_loader, max_length=model_params['max_o_len'])
        header = f"[ {datadir} ]"
        print(f"{header:<20} Threshold: 0.9 | Test Loss: {t_l:.4f} | Test chrF: {t_chrF:.4f} | Test BLEU: {t_bleu:.4f}")
        del model, ts_loader  # free GPU memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
