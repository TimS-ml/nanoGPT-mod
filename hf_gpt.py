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

from utils import *; from boring_utils.utils import *
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
    def __init__(self, config, scale_init=True):
        super().__init__()
        self.config = config
        self.scale_init = scale_init

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # https://www.youtube.com/watch?v=l8pRSuU81PU&t=3974s parameter sharing wte and lm_head
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        '''
        1/sqrt(768) = 0.036 and 1/sqrt(1600) = 0.025
        so the value in gpt2 paper 0.02 is reasonable
        '''
        if isinstance(module, nn.Linear):
            std = 0.02
            if self.scale_init:
                # '2 *' is because the two residual connections in the Block:
                # attn and mlp
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
        # preview the decay and nodecay layers
        tmp_decay_params = []
        tmp_nodecay_params = []
        for n, p in param_dict.items():
            if p.dim() >= 2 and len(tmp_decay_params) < 3:
                tmp_decay_params.append(n)
            elif p.dim() < 2 and len(tmp_nodecay_params) < 3:
                tmp_nodecay_params.append(n)
            if len(tmp_decay_params) >=3 and len(tmp_nodecay_params) >= 3:
                break
        cprint(tmp_decay_params, tmp_nodecay_params)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
    
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        cprint(len(decay_params), num_decay_params)
        cprint(len(nodecay_params), num_nodecay_params)
    
        # Create AdamW optimizer and use the fused version if it is available
        # NOTE: fused is only available on CUDA devices, but is much faster
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        cprint(f"using fused AdamW: {use_fused}")
    
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        '''https://youtu.be/l8pRSuU81PU?t=1830

        I insist using pytorch's MHA instead of HF. So I need a key_mapping dict.
        '''
        assert model_type in {'distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'distilgpt2':   dict(n_layer=6, n_head=12, n_embd=768),
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args.update(vocab_size=50257, block_size=1024, bias=True)

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        key_mapping = {
            'transformer.wte.weight': 'transformer.wte.weight',
            'transformer.wpe.weight': 'transformer.wpe.weight',
            'transformer.ln_f.weight': 'transformer.ln_f.weight',
            'transformer.ln_f.bias': 'transformer.ln_f.bias',
            'lm_head.weight': 'lm_head.weight',
        }

        for i in range(config.n_layer):
            layer_mapping = {
                f'transformer.h.{i}.ln_1.weight': f'transformer.h.{i}.ln_1.weight',
                f'transformer.h.{i}.ln_1.bias': f'transformer.h.{i}.ln_1.bias',
                f'transformer.h.{i}.ln_2.weight': f'transformer.h.{i}.ln_2.weight',
                f'transformer.h.{i}.ln_2.bias': f'transformer.h.{i}.ln_2.bias',
                f'transformer.h.{i}.mlp.c_fc.weight': f'transformer.h.{i}.mlp.c_fc.weight',
                f'transformer.h.{i}.mlp.c_fc.bias': f'transformer.h.{i}.mlp.c_fc.bias',
                f'transformer.h.{i}.mlp.c_proj.weight': f'transformer.h.{i}.mlp.c_proj.weight',
                f'transformer.h.{i}.mlp.c_proj.bias': f'transformer.h.{i}.mlp.c_proj.bias',
            }
            key_mapping.update(layer_mapping)

            # copy attn weights
            with torch.no_grad():
                c_attn_weight = model_hf.state_dict()[f'transformer.h.{i}.attn.c_attn.weight'].t()  # (3*n_embd, n_embd)
                c_attn_bias = model_hf.state_dict()[f'transformer.h.{i}.attn.c_attn.bias']  # (3*n_embd,)
                model.transformer.h[i].attn.in_proj_weight.copy_(c_attn_weight)
                model.transformer.h[i].attn.in_proj_bias.copy_(c_attn_bias)
                out_proj_weight = model_hf.state_dict()[f'transformer.h.{i}.attn.c_proj.weight'].t()
                out_proj_bias = model_hf.state_dict()[f'transformer.h.{i}.attn.c_proj.bias']
                model.transformer.h[i].attn.out_proj.weight.copy_(out_proj_weight)
                model.transformer.h[i].attn.out_proj.bias.copy_(out_proj_bias)

        for k in key_mapping:
            if k in model_hf.state_dict():
                if 'mlp' in k and k.endswith('.weight'):
                    model.state_dict()[key_mapping[k]].copy_(model_hf.state_dict()[k].t())
                else:
                    model.state_dict()[key_mapping[k]].copy_(model_hf.state_dict()[k])

        return model
