'''
A GPT2 implementation that can load HF checkpoints. GPT2 Small only.
'''

# %%
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from einops import rearrange, repeat, reduce

# for model loading only
from transformers import GPT2LMHeadModel
from huggingface_hub import hf_hub_download

from typing import Optional, Tuple, Union, List, Any, Generator, Type, Callable
from jaxtyping import Float, Bool

from boring_utils.utils import get_device, cprint, tprint

device = get_device()

# %%
class CasualSelfAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, max_seq_len: int = 1024, bias: bool = True):
        super().__init__()
        assert embedding_dim % num_heads == 0, f"n_embed {embedding_dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_size = embedding_dim // num_heads

        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)  # qkv projection
        self.c_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)  # output projection

        self.register_buffer(
                "mask", 
                torch.tril(torch.ones(max_seq_len, max_seq_len))
                    .view(1, 1, max_seq_len, max_seq_len))  # extend dims to 4

    def forward(
            self, 
            x: Float[Tensor, "batch seq_len embedding_dim"],
            mask: Optional[Bool[Tensor, "batch seq_len seq_len"]] = None,
            cache: Optional[Tuple[Tensor, Tensor]] = None
        ) -> Tuple[Float[Tensor, "batch seq_len embedding_dim"], Tuple[Tensor, Tensor]]:
        batch, seq_len, embedding_dim = x.shape

        # ["batch, seq_len, embedding_dim"] -> ["batch, seq_len, (3 * embedding_dim)"]
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embedding_dim, dim=-1)  # split at the last dim

        # embedding_dim = num_heads * head_dim
        # put seq_len and the head_dim together
        q, k, v = map(lambda t: rearrange(t, 'batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim', num_heads = self.num_heads), (q, k, v))

        if cache is not None:
            key_cache, value_cache = cache
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)

        norm_factor = 1.0 / np.sqrt(k.size(-1))  # k.size(-1) is the head_dim
        attn = (q @ k.transpose(-2, -1)) * norm_factor
        if mask is None:
            attn = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        else:
            mask = mask.bool()
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # attn: [batch, num_heads, seq_len, seq_len]
        # v:    [batch, num_heads, seq_len, head_dim]
        # y:    [batch, num_heads, seq_len, head_dim]
        y = attn @ v
        y = rearrange(y, 'batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)')
        return self.c_proj(y), (k, v)  # [batch, seq_len, embedding_dim]

# %%
class CasualSelfAttention_alternative(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, max_seq_len: int = 1024, bias: bool = True):
        super().__init__()
        assert embedding_dim % num_heads == 0, f"n_embed {embedding_dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_size = embedding_dim // num_heads

        # self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.transformer.heads = nn.ModuleList([
            nn.ModuleDict({
                'key': nn.Linear(embedding_dim, self.head_size, bias=bias),
                'query': nn.Linear(embedding_dim, self.head_size, bias=bias), 
                'value': nn.Linear(embedding_dim, self.head_size, bias=bias)
            }) for _ in range(num_heads)
        ])
        self.c_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)  # output projection

        self.register_buffer(
                "mask", 
                torch.tril(torch.ones(max_seq_len, max_seq_len))
                    .view(1, 1, max_seq_len, max_seq_len))  # extend dims to 4

    def forward(
            self, 
            x: Float[Tensor, "batch seq_len embedding_dim"]
        ) -> Float[Tensor, "batch seq_len embedding_dim"]:
        batch, seq_len, embedding_dim = x.shape

        # cat([batch, seq_len, head_dim] x num_heads) -> [batch, seq_len, num_heads * head_dim]
        q = torch.cat([h['query'](x) for h in self.transformer.heads], dim=-1)
        k = torch.cat([h['key'](x) for h in self.transformer.heads], dim=-1)
        v = torch.cat([h['value'](x) for h in self.transformer.heads], dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim', num_heads = self.num_heads), (q, k, v))

        norm_factor = 1.0 / np.sqrt(k.size(-1))  # k.size(-1) is the head_dim
        attn = (q @ k.transpose(-2, -1)) * norm_factor
        attn = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # attn: [batch, num_heads, seq_len, seq_len]
        # v:    [batch, num_heads, seq_len, head_dim]
        # y:    [batch, num_heads, seq_len, head_dim]
        y = attn @ v
        y = rearrange(y, 'batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)')
        return self.c_proj(y)  # [batch, seq_len, embedding_dim]

# %%
class GELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class QuickGELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)

# %%
class FFN(nn.Module):
    def __init__(self, embedding_dim: int, bias: bool = True):
        super().__init__()
        hidden_dim = embedding_dim * 4
        self.c_fc = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        # self.gelu = nn.GELU(approximate='tanh')
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(hidden_dim, embedding_dim, bias=bias)

    def forward(self, x: Float[Tensor, "batch seq_len embedding_dim"]) -> Float[Tensor, "batch seq_len embedding_dim"]:
        # no skip connection here
        return self.c_proj(self.gelu(self.c_fc(x)))

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embedding_dim))  # scaling (gamma)
        self.bias = nn.Parameter(torch.zeros(embedding_dim))  # offset (beta)
        self.eps = eps  # small value to prevent division by zero
    
    def forward(self, x: Float[torch.Tensor, "batch seq_len embedding_dim"]) -> Float[torch.Tensor, "batch seq_len embedding_dim"]:
        mean = x.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [batch, seq_len, 1]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # [batch, seq_len, embedding_dim]
        return self.weight * x_norm + self.bias

# %%
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Subtract max value for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    
    # Calculate denominator (sum) and normalize
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


class CrossEntropyLoss(nn.Module):
    """
    loss = -sum(y_true * log(y_pred))
    """
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        logits: torch.Tensor,   # Raw logits from model, shape (batch_size, num_classes)
        targets: torch.Tensor,  # Target labels, shape (batch_size,)
    ) -> torch.Tensor:
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, num_classes)
        
        # Gather log probabilities of target classes
        # gather operation collects values from log_probs at positions specified by targets
        target_log_probs = log_probs.gather(
            dim=-1,
            index=targets.unsqueeze(-1)
        ).squeeze(-1)        

        loss = -target_log_probs.mean()
        return loss


# x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# probs = softmax(x)
# print(f"Softmax output: {probs}")
# print("Sum of probabilities:", probs.sum(dim=-1))
    
# criterion = CrossEntropyLoss()
# logits = torch.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])  # (2, 3)
# targets = torch.tensor([0, 1])
# loss = criterion(logits, targets)
# print("\nCross Entropy Loss:", loss.item())

# %%
class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, max_seq_len: int = 1024, bias: bool = True):
        super().__init__()
        # self.ln_1 = nn.LayerNorm(embedding_dim, bias=bias)  # norm on the last dim
        # self.ln_2 = nn.LayerNorm(embedding_dim, bias=bias)
        self.ln_1 = LayerNorm(embedding_dim)  # norm on the last dim
        self.ln_2 = LayerNorm(embedding_dim)
        self.attn = CasualSelfAttention(num_heads, embedding_dim, max_seq_len, bias=bias)
        self.mlp = FFN(embedding_dim, bias=bias)
    
    def forward(
            self, 
            x: Float[Tensor, "batch seq_len embedding_dim"],
            mask: Optional[Bool[Tensor, "batch seq_len seq_len"]] = None,
            cache: Optional[Tuple[Tensor, Tensor]] = None
        ) -> Tuple[Float[Tensor, "batch seq_len embedding_dim"], Tuple[Tensor, Tensor]]:
        # skip connection, pre-layer norm
        # x = x + self.attn(self.ln_1(x))
        att, cache = self.attn(self.ln_1(x), mask=mask, cache=cache)
        x = x + att
        x = x + self.mlp(self.ln_2(x))
        return x, cache

# %%
class GPT(nn.Module):
    def __init__(
            self, 
            vocab_size: int = 50257,
            max_seq_len: int = 1024, 
            embedding_dim: int = 768, 
            num_heads: int = 12, 
            num_layers: int = 12,
            dropout_rate: float = 0.0,
            bias: bool = True
        ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embedding_dim),
            wpe = nn.Embedding(max_seq_len, embedding_dim),
            drop = nn.Dropout(dropout_rate),
            h = nn.ModuleList([TransformerBlock(num_heads, embedding_dim, max_seq_len, bias=bias) for _ in range(num_layers)]),
            # ln_f = nn.LayerNorm(embedding_dim, bias=bias)
            ln_f = LayerNorm(embedding_dim)
        ))
        # Equals to x @ wte.weight.T
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def _forward_transformer_blocks(
            self, 
            x: Float[Tensor, "batch seq_len embedding_dim"],
            mask: Optional[Bool[Tensor, "batch seq_len seq_len"]] = None,
            cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
            build_cache: bool = False
        ) -> Tuple[Float[Tensor, "batch seq_len embedding_dim"], Optional[Tuple[Tensor, Tensor]]]:
        x = self.transformer.drop(x)
        kv_cache = []
        
        if cache is not None:
            for i in range(len(cache)):
                x, cache[i] = self.transformer.h[i](x, mask=None, cache=cache[i])
        else:
            for block in self.transformer.h:
                x, curr_cache = block(x, mask=mask)
                if build_cache:
                    kv_cache.append(curr_cache)
                    
        x = self.transformer.ln_f(x)
        return x, kv_cache if build_cache else cache

    def forward(
            self, 
            x: Float[Tensor, "batch seq_len"],
            mask: Optional[Bool[Tensor, "batch seq_len seq_len"]] = None,
            cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
            build_cache: bool = False
        ) -> Tuple[Float[Tensor, "batch seq_len vocab_size"], Optional[Tuple[Tensor, Tensor]]]:
        batch, seq_len = x.shape
        assert seq_len <= self.max_seq_len, f"input length {seq_len} is longer than max seq length {self.max_seq_len}"

        pos = torch.arange(0, seq_len, device=x.device)
        pos_emb = self.transformer.wpe(pos)  # [seq_len, embedding_dim]
        tok_emb = self.transformer.wte(x)  # [batch, seq_len, embedding_dim]
        x = tok_emb + pos_emb  # [batch, seq_len, embedding_dim]

        x, kv_cache = self._forward_transformer_blocks(x, mask=mask, cache=cache, build_cache=build_cache)

        # Same as: logits = x @ self.wte.weight.T
        logits = self.lm_head(x) # [batch, seq_len, vocab_size]

        if build_cache:
            return logits, kv_cache
        return logits, None

    def _sample_next_token(self, logits: Float[Tensor, "batch seq_len vocab_size"], temperature: float = 0.8) -> Float[Tensor, "batch 1"]:
        logits = logits[:, -1, :]  # [batch, vocab_size]
        probs = torch.softmax(logits * (1 / temperature), dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)  # [batch, 1]
        xcol = torch.gather(topk_indices, -1, ix)  # [batch, 1]
        return xcol

    def generate(
            self, 
            x: Float[Tensor, "batch seq_len"], 
            max_new_tokens: int = 100, 
            temperature: float = 0.8
        ) -> Generator[
            Float[Tensor, "batch 1"],  # yield
            None,  # generator.send()
            List[Float[Tensor, "batch 1"]]  # generator.throw()
        ]:
        """
        # Method 1: Get tokens one by one using a for loop
        for token in model.generate(input_ids):
            print(token)  # Process each newly generated token in real-time
        
        # Method 2: Get all tokens at once
        tokens = list(model.generate(input_ids))
        """
        logits, cache = self.forward(x, build_cache=True)
        
        tokens = []
        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(logits, temperature)
            yield next_token
            
            tokens.append(next_token)
            
            # forward pass only for the new token
            tok_emb = self.transformer.wte(next_token)  # [batch, 1, embedding_dim]
            pos_emb = self.transformer.wpe(
                torch.tensor([x.size(1)], dtype=torch.long, device=x.device)
            ).unsqueeze(0)  # [1, 1, embedding_dim]
            
            hidden = tok_emb + pos_emb
            
            hidden, cache = self._forward_transformer_blocks(hidden, cache=cache)
            logits = self.lm_head(hidden)
            
            x = torch.cat((x, next_token), dim=1)
            
        del cache
        torch.cuda.empty_cache()
        
        return tokens    
    
    @classmethod
    def from_pretrained(cls, model: Optional[Union[None, "GPT", Type["GPT"]]] = None, rlhf: bool = False, sft: bool = False):
        '''https://youtu.be/l8pRSuU81PU?t=1830
        '''
        if model is None: 
            model = cls(vocab_size=50260) if (rlhf or sft) else cls()
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.mask')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        if sft:
            print("Model type: SFT GPT2")
            model_hf = GPT2LMHeadModel.from_pretrained('vicgalle/gpt2-alpaca-gpt4')
        elif rlhf:
            print("Model type: RLHF GPT2")
            model_hf = GPT2LMHeadModel.from_pretrained('jtatman/gpt2-open-instruct-v1-Anthropic-hh-rlhf')
        else:
            print("Model type: Regular GPT2")
            model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.mask')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # print('hf:   ', [k for k in sd_keys_hf if "h.0" in k])
        # print('mine: ', [k for k in sd_keys if "h.0" in k])

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"{k} shape mismatch: {sd_hf[k].shape[::-1]} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"{k} shape mismatch: {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# model = GPT.from_pretrained()
# model.eval()
# model.to(device)

# %%
# NOTE: no kv cache and streaming decode here
def generate_text_simple(
    tokenizer: Any, 
    question: str, 
    model: GPT = model, 
    num_attempt: int = 3,  # num_attempt = batch
    max_length: int = 100
):
    # tokenizer encode
    tokens = tokenizer.encode(question)  # [seq_len]
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_attempt, 1)  # [num_attempt, seq_len]
    x = tokens.to(device)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)  # [batch, curr_seq_len, vocab_size]

        # take the logits at the last position
        logits = logits[:, -1, :]  # [batch, vocab_size]

        # get the probabilities
        probs = F.softmax(logits, dim=-1)

        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        # turn to zero for all indices below the top-k
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        # [Multinomial distribution - Wikipedia](https://en.wikipedia.org/wiki/Multinomial_distribution)
        ix = torch.multinomial(topk_probs, 1)  # [batch, 1]

        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # [batch, 1]

        # append to the sequence
        x = torch.cat((x, xcol), dim=1)  # [batch, curr_seq_len + 1]

    # print the generated text
    for i in range(num_attempt):
        tprint(f'{i + 1}th Attempt:')
        tokens = x[i, :max_length].tolist()

        # tokenizer decode
        decoded = tokenizer.decode(tokens)
        print(f"> {decoded}")
        print()

# %%
def generate_text(
    tokenizer: Any, 
    question: str, 
    model: Any, 
    num_attempt: int = 3,  # num_attempt = batch
    max_length: int = 100,
    temperature: float = 1.0  # default
):
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py

    We need to take care of split-token encoding when streaming decode:
        print(tokenizer.decode([447, 247]))  # ’
        print(tokenizer.decode([447]).encode('utf-8'))  # �
        print(tokenizer.decode([171, 120, 253]))  # ？
    """
    special_sequences = {
        (447, 246): "‘",
        (447, 247): "’",
        (564, 250): "“",
        (447, 251): "”",
    }

    # BOS token ID = 50256
    tokens = tokenizer.encode(question) if question else [50256]
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_attempt, 1)  # [num_attempt, seq_len]
    x = tokens.to(device)

    for i in range(num_attempt):
        tprint(f'{i + 1}th Attempt:', c='yellow')
        curr_x = x[i: i+1]  # [1, seq_len]

        # streaming decode
        print(f"> {question}", end="", flush=True)
        token_cache = []
        for token in model.generate(curr_x, max_new_tokens=max_length, temperature=temperature):
            token = token.item()
            token_cache.append(token)
            
            decoded_text = ""
            for seq, char in special_sequences.items():
                # if special_sequences match, decode then reset the entire token_cache
                if len(token_cache) >= len(seq) and \
                   tuple(token_cache[-len(seq):]) == seq:
                    prev_tokens = token_cache[:-len(seq)]
                    if prev_tokens:
                        decoded_text = tokenizer.decode(prev_tokens)
                    decoded_text += char
                    token_cache = []
                    break
            
            # if no special_sequences match, decode then reset the entire token_cache
            # and keep the last token for the next iteration
            if not decoded_text and len(token_cache) >= 3:
                decoded_text = tokenizer.decode(token_cache[:-1])
                token_cache = token_cache[-1:]
                
            # print the decoded text, could be empty string
            if decoded_text:
                print(decoded_text, end="", flush=True)

        # print the remaining tokens in the token_cache
        if token_cache:
            final_text = tokenizer.decode(token_cache)
            if final_text:
                print(final_text, end="", flush=True)
        print()