import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
from transformers import PreTrainedTokenizerFast
import os
from sklearn.model_selection import train_test_split

# model init
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
device = torch.device("cpu")
state = torch.load('./models/wreckgar-4.pt', map_location=device)
model.load_state_dict(state)


def predictNext(text, k=20):
    tokens_tensor = torch.tensor([tokenizer.encode(text)]).to(device)

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # Get the predicted next sub-word
    probs = predictions[0, -1, :]
    top_next = [tokenizer.decode(i.item()).strip() for i in probs.topk(k)[1]]
    return top_next
