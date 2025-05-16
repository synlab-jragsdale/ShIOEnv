from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class BashDataset(Dataset):
    """
    T5/CodeT5 dataset for shell‑command -> diff/output generation.

    data : list, Raw JSON records - ndjson file.
    tokenizer : PreTrainedTokenizer
    i_len : int, default 512
    o_len : int, default 512
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        i_len: int = 512,
        o_len: int = 512,
        verbose: bool = False
    ) -> None:
        self.tokenizer = tokenizer
        self.max_i_len = i_len
        self.max_o_len = o_len

        # Pre‑init tensors
        self._input_ids: List[torch.Tensor] = []
        self._input_masks: List[torch.Tensor] = []
        self._target_ids: List[torch.Tensor] = []

        self._prep_data(data)
        if verbose:
            print(f"{len(self)} samples loaded (pre‑tokenised)")

    def _prep_data(self, raw_data: List[Dict]) -> None:
        for rec in raw_data:
            # clean source & target strings: concatenated output/diff
            clean_input = ";".join(rec["input"].split(";")[1:])
            combined_out = rec["output"] + rec["context_value"]

            # pre-tokenize
            src = self.tokenizer.encode_plus(
                clean_input,
                max_length=self.max_i_len,
                truncation=True,
                padding="max_length",
            )
            tgt = self.tokenizer.encode_plus(
                combined_out,
                max_length=self.max_o_len,
                truncation=True,
                padding="max_length",
            )

            # Stash tensors
            self._input_ids.append(torch.tensor(src["input_ids"], dtype=torch.long))
            self._input_masks.append(torch.tensor(src["attention_mask"], dtype=torch.long))
            self._target_ids.append(torch.tensor(tgt["input_ids"], dtype=torch.long))

    # dataloader get funs
    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "source_ids": self._input_ids[idx],
            "source_mask": self._input_masks[idx],
            "target_ids": self._target_ids[idx],
        }
