import os
import regex as re
import json
import requests
from collections import OrderedDict
from typing import Optional, List
from boring_utils.utils import cprint, tprint

# %% [markdown]
# # BPE (Byte Pair Encoding)
# 
# ```
# r"""'s|'t|'re|'ve|'m|'ll|'d  Match common English contractions like 's, 't, 're, 've, 'm, 'll, 'd
# \p{L}+                       Match any sequence of Unicode letter characters (like English words)
# \p{N}+                       Match any sequence of Unicode numeric characters (like 123, 3.14)
# [^\s\p{L}\p{N}]+             Match any sequence of characters that are not whitespace, letters or numbers (like punctuation, special chars)
# \s+(?!\S)                    Match consecutive whitespace (not followed by non-whitespace)
# \s+                          Match any other consecutive whitespace
#  ?                           Match an optional space
# """
# ```

# %%
# import tiktoken
# tokenizer = tiktoken.get_encoding('gpt2')

# %% [markdown]
# ## OpenAI's Byte Encoder
# In utf-8:
# - 0-31 are control characters, e.g. \x00 is null, \x01 is start of heading, \x09 is tab etc.
# - 32-127 are basic Latin letters, numbers and some punctuation marks
# - 128-255 are extended ASCII codes, including accented letters and special characters

# %%
def bytes_to_unicode():
    """
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
    character that represents it visually.
    """
    # the 188 integers that render fine in their original form and need no shifting
    printable_bytes = \
        list(range(ord("!"), ord("~")+1)) + \
        list(range(ord("¡"), ord("¬")+1)) + \
        list(range(ord("®"), ord("ÿ")+1))

    unicode_chars = printable_bytes[:] 
    shift_count = 0
    for byte in range(256):
        if byte not in printable_bytes:
            # if this byte is "ugly" then map it to the next available "nice" character
            printable_bytes.append(byte)
            unicode_chars.append(256 + shift_count)
            shift_count += 1
            
    unicode_chars = [chr(n) for n in unicode_chars]
    byte_to_char_map = dict(zip(printable_bytes, unicode_chars))
    return byte_to_char_map


# NOTE: Don't be fooled by the printed output, the dict should be {b'\x21': '!', b'\x22': '"', ...} instead of {33: '!', 34: '"', ...}
# cprint(bytes_to_unicode()[ord(b'\x21')])
# cprint(bytes_to_unicode()[33])

# %%
# cprint(bytes_to_unicode(), use_pprint=False)

# %%
class BPETokenizer:
    """
    https://tiktokenizer.vercel.app/?model=gpt2
    """
    def __init__(self, encoder: dict = None, bpe_merges: dict = None):
        # encoder: map bytes to unicode characters
        # decoder: inverse of encoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}

        # encoder: bpe token to index, json dict
        # {... "clud": 758, "tern": 759, "\u0120know": 760 ...}
        # decoder: index to bpe token
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}

        # bpe merge list that defines the bpe "tree"
        # {... Ġre claimed, Ġinteresting ly, × ©, rom y, J M, ĠEnhance ment, ...}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

        # ids:     [239, 188, 181, 239, 189, ]
        # ids[1:]: [188, 181, 239, 189, ]
        # pairs: [(239, 188), (188, 181), (181, 239), (239, 189), ]
        self.get_pairs = lambda word: set(zip(word, word[1:]))

    def decode(self, ids: List[int]) -> str:
        if not ids: return ""
        tokens = [self.decoder[i] for i in ids]
        tokens_flat = ''.join(tokens)

        # recovering 'Ġ' -> ' '
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        return tokens_bytes.decode('utf-8', errors='replace')

    def bpe_merge(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs: return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))

            if bigram not in self.bpe_ranks: break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):

                # find the next occurence of first in the sequence of current words
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # if this occurence is also followed by second, then merge them into one
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        # concat all words into a string, and use ' ' as the separator. Note that
        # by now all characters have been byte encoded, guaranteeing that ' ' is
        # not used in the actual data and is a 'special' delimiter character
        word = ' '.join(word)

        # cache the result and return
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_idx = []
        # pre-tokenize the input text into a list of string tokens, this is the minimum unit of tokenization
        # input: "Hello've world123!!!?    "
        # output: ['Hello', "'ve", ' world', '123', '!!!', '?', '    ']
        tokens = re.findall(self.gpt2pat, text)

        for token in tokens:
            # char to bytes
            token_bytes = token.encode('utf-8')

            # apply the openai byte encoder to the token, ' word' -> 'Ġword'
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)

            # perform all the applicable bpe merges according to self.bpe_ranks
            # 'interestingly' -> 'interest' + 'ingly'
            token_merged = self.bpe_merge(token_translated).split(' ')

            # translate all bpe tokens to integers
            # 'interest' + 'ingly' -> [9446, 4420]
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]

            # extend our running list of all output integers
            bpe_idx.extend(token_ix)
        return bpe_idx

    @classmethod
    def from_pretrained(cls, rlhf_token=False):
        data_dir = './checkpoint/gpt2_tokenizer/'
        os.makedirs(data_dir, exist_ok=True)

        # load encoder.json that has the raw mappings from token -> bpe index
        encoder_path = os.path.join(data_dir, 'encoder.json')
        if not os.path.isfile(encoder_path):
            encoder_remote_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
            response = requests.get(encoder_remote_url)
            open(encoder_path, "wb").write(response.content)
        with open(encoder_path, 'r') as f:
            encoder = json.load(f)
        assert len(encoder) == 50257  # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token

        if rlhf_token:
            encoder["### End"] = 50257
            encoder["### Instruction:"] = 50258
            encoder["### Response:\n"] = 50259

        # load vocab.bpe that contains the bpe merges, i.e. the bpe tree structure
        vocab_path = os.path.join(data_dir, 'vocab.bpe')
        if not os.path.isfile(vocab_path):
            vocab_remote_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
            response = requests.get(vocab_remote_url)
            open(vocab_path, "wb").write(response.content)
        with open(vocab_path, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        assert len(bpe_merges) == 50000  # 50,000 merged tokens

        # construct the Encoder object and return
        enc = BPETokenizer(encoder, bpe_merges)
        return enc


# tokenizer_2 = BPETokenizer.from_pretrained()

# %% [markdown]
# ## BPE Training
# 
# ```python
# def get_stats(ids):
#     counts = {}
#     # Pythonic way to iterate consecutive elements
#     # ids:     [239, 188, 181, 239, 189, ]
#     # ids[1:]: [188, 181, 239, 189, ]
#     # pairs: [(239, 188), (188, 181), (181, 239), (239, 189), ]
#     for pair in zip(ids, ids[1:]):
#         counts[pair] = counts.get(pair, 0) + 1
#     return counts
# 
# def single_merge(ids, pair, idx):
#     # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
#     # single_merge([5, 6, 6, 7, 9, 1], (6, 7), 99) -> [5, 6, 99, 9, 1]
#     newids = []
#     i = 0
#     while i < len(ids):
#         # if we are not at the very last position AND the pair matches, replace it
#         if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
#             newids.append(idx)
#             i += 2
#         else:
#             newids.append(ids[i])
#             i += 1
#     return newids
# 
# # top_pair = max(stats, key=stats.get)
# # tokens2 = merge(tokens, top_pair, 256)
# ```
