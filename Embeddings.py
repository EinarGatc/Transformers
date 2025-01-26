import torch
import torch.nn as nn 
import random
from transformers import BertTokenizer, BertModel, BertConfig
import math

random_seed = 42
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class InputEmbedding(nn.Module):
    def __init__(self, embed_size:int):
        super(InputEmbedding, self).__init__()
        self.embed_size = embed_size

    def getEmbedding(self, text: list[str]):
        encoding = tokenizer.batch_encode_plus(
            text,
            padding=True,              # Pad to the maximum sequence length
            truncation=True,           # Truncate to the maximum sequence length if necessary
            return_tensors='pt',      # Return PyTorch tensors
            add_special_tokens=True    # Add special tokens CLS and SEP
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']  # Attention mask

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state

        return word_embeddings
        


if __name__ == "__main__":
    text = ["&quot;GeeksforGeeks is a computer science portal&quot"]
    o1 = InputEmbedding(512)
    word_embeddings = o1.getEmbedding(text)
    print(word_embeddings.shape)