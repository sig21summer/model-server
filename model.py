from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizerFast
import torch

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
