import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from transformers import BertTokenizer, BertModel, BertConfig
import math

random_seed = 42
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_model // num_heads

        assert d_model * num_heads == self.d_head
        
        self.attention_heads = [Attention(self.d_model, self.d_head) for _ in num_heads]

    def forward(self, x):
        combined = []

        return combined
    
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_model // num_heads

        assert d_model * num_heads == self.d_head
        
        self.attention_heads = [Attention(self.d_model, self.d_head) for _ in num_heads]

    def forward(self, x):
        combined = []

        return combined


class MaskedAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        
        self.query = nn.Linear(d_model, d_head)
        self.key = nn.Linear(d_model, d_head)
        self.value = nn.Linear(d_model, d_head)

        self.mask = self.create_mask(d_head)

    def create_mask(size):
        # Create lower triangular matrix (subsequent positions masked)
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.masked_fill(mask == True, float('-inf'))
    
    def forward(self, x):
        q_p = x @ self.query
        k_p = x @ self.key
        v_p = x @ self.value

        attention = q_p @ k_p
        attention = attention / torch.sqrt(self.d_head)
        attention = attention + self.mask
        attention = F.softmax(attention, dim=-1)
        attention = attention @ v_p

        return attention
    
    def create_mask(size):
        # Create lower triangular matrix (subsequent positions masked)
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.masked_fill(mask == True, float('-inf'))

class Attention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        
        self.query = nn.Linear(d_model, d_head)
        self.key = nn.Linear(d_model, d_head)
        self.value = nn.Linear(d_model, d_head)

    def forward(self, x):
        q_p = x @ self.query
        k_p = x @ self.key
        v_p = x @ self.value

        attention = q_p @ k_p
        attention = attention / torch.sqrt(self.d_head)
        attention = F.softmax(attention, dim=-1)
        attention = attention @ v_p

        return attention

class TranformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm()
        self.ffn = FeedForward(d_model, d_model)
        self.norm2 = nn.LayerNorm()
    
    def forward(self, x):
        attention = self.attention(x)
        attention = attention + x
        attention = self.norm1(attention)
        
        final_scores = self.ffn(attention)
        final_scores = final_scores + attention
        final_scores = self.norm2(final_scores)
        
        return final_scores



class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

class FeedForward(nn.Module):
   def __init__(self, d_model, d_ff):
       super().__init__()
       self.linear1 = nn.Linear(d_model, d_ff)
       self.linear2 = nn.Linear(d_ff, d_model)
       
   def forward(self, x):
       return self.linear2(F.relu(self.linear1(x)))