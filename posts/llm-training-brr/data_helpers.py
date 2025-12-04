import os, math
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


# caching, dataset
HF_HOME = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "hub"))

HF_DS_CACHE = os.environ["HF_DATASETS_CACHE"]
HF_HUB_CACHE = os.environ["TRANSFORMERS_CACHE"]
print(f"HF datasets cache: {HF_DS_CACHE}\nHF hub cache:      {HF_HUB_CACHE}")

# slices
train_split = "train[:5%]"
val_split   = "validation[:2%]"

PACKED_TRAIN_PATH = "tinystories_train_packed.pt"
PACKED_VAL_PATH   = "tinystories_val_packed.pt"

# --------------------------
# Data helpers
# --------------------------
def to_packed_ids(tok_ds, context_len):
    ids_list = tok_ds["input_ids"]
    total_len = sum(len(a) for a in ids_list)
    ids = torch.empty(total_len, dtype=torch.long)
    pos = 0
    for arr in ids_list:
        n = len(arr)
        ids[pos:pos+n] = torch.as_tensor(arr, dtype=torch.long)
        pos += n
    n = (len(ids) - 1) // context_len
    if n <= 0:
        raise RuntimeError("Not enough tokens; enlarge dataset or reduce context_len.")
    x = ids[: n * context_len].view(n, context_len)
    y = ids[1 : n * context_len + 1].view(n, context_len)
    return x, y

class DeviceDataset(Dataset):
    """
    Simple dataset that keeps x/y tensors on the target device permanently.
    """
    def __init__(self, x, y, device):
        if x.shape != y.shape:
            raise ValueError(f"x and y must have matching shape, got {x.shape} vs {y.shape}")
        self.device = torch.device(device)
        self.x = x.contiguous().to(self.device)
        self.y = y.contiguous().to(self.device)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class DeviceBatcher:
    """
    Lightweight iterable for batching an in-memory DeviceDataset.
    """
    def __init__(self, dataset: DeviceDataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            idx = torch.randperm(n, device=self.dataset.device)
        else:
            idx = torch.arange(n, device=self.dataset.device)

        for start in range(0, n, self.batch_size):
            sl = idx[start:start + self.batch_size]
            yield (
                self.dataset.x.index_select(0, sl),
                self.dataset.y.index_select(0, sl),
            )


def build_device_batches(x_train, y_train, x_val, y_val, batch_size, device):
    train_ds = DeviceDataset(x_train, y_train, device)
    val_ds   = DeviceDataset(x_val,   y_val,   device)
    train_batches = DeviceBatcher(train_ds, batch_size=batch_size, shuffle=True)
    val_batches   = DeviceBatcher(val_ds,   batch_size=batch_size, shuffle=False)
    return train_batches, val_batches

def load_or_build_packed(context_len):
    print("Loading TinyStories… (will reuse local cache if present)")
    ds_train = load_dataset(
        "roneneldan/TinyStories",
        split=train_split, cache_dir=HF_DS_CACHE,
        download_mode="reuse_cache_if_exists",
    )
    ds_val = load_dataset(
        "roneneldan/TinyStories",
        split=val_split, cache_dir=HF_DS_CACHE,
        download_mode="reuse_cache_if_exists",
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=HF_HUB_CACHE)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e9)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], add_special_tokens=True, truncation=False)

    print("Tokenizing...")
    tok_train = ds_train.map(
        tokenize_batch, batched=True,
        remove_columns=ds_train.column_names
    )
    tok_val = ds_val.map(
        tokenize_batch, batched=True,
        remove_columns=ds_val.column_names
    )

    if os.path.exists(PACKED_TRAIN_PATH) and os.path.exists(PACKED_VAL_PATH):
        pk_train = torch.load(PACKED_TRAIN_PATH, map_location="cpu")
        pk_val   = torch.load(PACKED_VAL_PATH,   map_location="cpu")
        x_train, y_train = pk_train["x"], pk_train["y"]
        x_val,   y_val   = pk_val["x"],   pk_val["y"]
    else:
        x_train, y_train = to_packed_ids(tok_train, context_len)
        x_val,   y_val   = to_packed_ids(tok_val,   context_len)
        torch.save({"x": x_train, "y": y_train}, PACKED_TRAIN_PATH)
        torch.save({"x": x_val,   "y": y_val},   PACKED_VAL_PATH)

    vocab_size = tokenizer.vocab_size
    return x_train, y_train, x_val, y_val, vocab_size, tokenizer
