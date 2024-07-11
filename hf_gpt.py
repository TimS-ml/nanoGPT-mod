'''
A GPT2 implementation that can load HF checkpoints
'''

import os
import numpy as np
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
# from transformers import GPT2LMHeadModel
import matplotlib.pyplot as plt 

from utils import *
from data_structure import add_to_class

# init_graph()
device = get_device()


# bias True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
@dataclass
class GPTConfig_small:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True

# vocab_size: int = 50304: GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
    
    def forward(self, x):
        x = self.c_fc(x)
        # x = F.gelu(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# class Block(nn.Module):
#     '''
#     Attn is the 'reduce', MLP is the 'map' (no cross token ops)
#     Pytorch MHA's input shape is: sequence length first (or batch_first=True)
#     '''
#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
#         self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
#         # self.attn = CasualSelfAttention(config)
#         self.attn = nn.MultiheadAttention(
#             config.n_embd, config.n_head)
#         self.mlp = MLP(config)
        
#         self.n_head = config.n_head
#         self.register_buffer(
#             "mask",
#             torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1).bool()
#         )

#     def forward(self, x):
#         B, T, C = x.size()
#         # mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
#         mask = self.mask[:T, :T]
#         x = x + self.attn(self.ln_1(x).transpose(0, 1), 
#                           self.ln_1(x).transpose(0, 1), 
#                           self.ln_1(x).transpose(0, 1), 
#                           attn_mask=mask)[0].transpose(0, 1)
#         x = x + self.mlp(self.ln_2(x))
#         return x


class Block(nn.Module):
    '''
    Attn is the 'reduce', MLP is the 'map' (no cross token ops)
    Pytorch MHA's input shape is: sequence length first (or batch_first=True)
    '''
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        # self.attn = CasualSelfAttention(config)
        self.attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True)
        self.mlp = MLP(config)
        
        self.n_head = config.n_head
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.size()
        # mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        mask = self.mask[:T, :T]
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # https://www.youtube.com/watch?v=l8pRSuU81PU&t=3974s parameter sharing wte and lm_head
        self.model.transformer.wte.weight = self.lm_head.weight
    
    def forward(self, idx, targets=None):
        # idx shape: (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"input length {T} is longer than block size {self.config.block_size}"
        # pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        pos = torch.arange(0, T, device=idx.device)  # shape: T
        pos_emb = self.transformer.wpe(pos)  # shape: (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # shape: (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # shape: (B, T, Vocab Size)

        if targets is None:
            return logits
        else:
            # logits.view(-1, logits.size(-1)): 
            # flatten: (B, T, Vocab Size) -> (B * T, Vocab Size) 
            loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        targets.view(-1)
                    )
            return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        '''https://youtu.be/l8pRSuU81PU?t=1830

        I insist using pytorch's MHA instead of HF. So I need a key_mapping dict.
        '''
        assert model_type in {'distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'distilgpt2':   dict(n_layer=6, n_head=12, n_embd=768),  # 84M params
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.mask')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.mask')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}, \n\n{sd_keys_hf}, \n\n{sd_keys}"

        # create a mapping from the Hugging Face model keys to your custom model keys
        key_mapping = {
            'transformer.wte.weight': 'transformer.wte.weight',
            'transformer.wpe.weight': 'transformer.wpe.weight',
            'transformer.ln_f.weight': 'transformer.ln_f.weight',
            'transformer.ln_f.bias': 'transformer.ln_f.bias',
            'lm_head.weight': 'lm_head.weight',
        }
        for i in range(config.n_layer):
            key_mapping.update({
                f'transformer.h.{i}.ln_1.weight': f'transformer.h.{i}.ln_1.weight',
                f'transformer.h.{i}.ln_1.bias': f'transformer.h.{i}.ln_1.bias',
                f'transformer.h.{i}.attn.c_attn.weight': f'transformer.h.{i}.attn.in_proj_weight',
                f'transformer.h.{i}.attn.c_attn.bias': f'transformer.h.{i}.attn.in_proj_bias',
                f'transformer.h.{i}.attn.c_proj.weight': f'transformer.h.{i}.attn.out_proj.weight',
                f'transformer.h.{i}.attn.c_proj.bias': f'transformer.h.{i}.attn.out_proj.bias',
                # f'transformer.h.{i}.attn.in_proj_weight': f'transformer.h.{i}.attn.c_attn.weight',
                # f'transformer.h.{i}.attn.in_proj_bias': f'transformer.h.{i}.attn.c_attn.bias',
                # f'transformer.h.{i}.attn.out_proj.weight': f'transformer.h.{i}.attn.c_proj.weight',
                # f'transformer.h.{i}.attn.out_proj.bias': f'transformer.h.{i}.attn.c_proj.bias',
                f'transformer.h.{i}.ln_2.weight': f'transformer.h.{i}.ln_2.weight',
                f'transformer.h.{i}.ln_2.bias': f'transformer.h.{i}.ln_2.bias',
                f'transformer.h.{i}.mlp.c_fc.weight': f'transformer.h.{i}.mlp.c_fc.weight',
                f'transformer.h.{i}.mlp.c_fc.bias': f'transformer.h.{i}.mlp.c_fc.bias',
                f'transformer.h.{i}.mlp.c_proj.weight': f'transformer.h.{i}.mlp.c_proj.weight',
                f'transformer.h.{i}.mlp.c_proj.bias': f'transformer.h.{i}.mlp.c_proj.bias',
            })

        # cprint("transformer.h.0.attn.c_attn.weight" in sd_keys_hf)
        # cprint("transformer.h.0.attn.c_attn.weight" in sd_keys)  # False, so we need key_mapping
        # print('hf:   ', [k for k in sd_keys_hf if "h.0" in k])
        # print('mine: ', [key_mapping[k] for k in sd_keys if "h.0" in k])
        print('hf:   ', [key_mapping[k] for k in sd_keys_hf if "h.0" in k])
        print('mine: ', [k for k in sd_keys if "h.0" in k])

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[key_mapping[k]].shape, f"mismatched keys: {sd_hf[k].shape}, {sd[key_mapping[k]].shape}"
                with torch.no_grad():
                    sd[key_mapping[k]].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[key_mapping[k]].shape
                with torch.no_grad():
                    sd[key_mapping[k]].copy_(sd_hf[k])

        return model


# num_return_sequences = 4
# max_length = 30

# # model = GPT.from_pretrained('distilgpt2')
# model = GPT.from_pretrained('gpt2')
# model.eval()
# model.to(device)
