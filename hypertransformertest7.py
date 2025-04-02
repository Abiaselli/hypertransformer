import os
import torch
import torch.nn as nn
import torch.fft
import logging
import math
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from transformers import PreTrainedTokenizerFast
import re
import torch.utils.checkpoint as checkpoint
import random
import pandas as pd
import intel_extension_for_pytorch as ipex

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Debug for CUDA
print(torch.__version__)
print(ipex.__version__)
for i in range(torch.xpu.device_count()):
    print(f'[{i}]: {torch.xpu.get_device_properties(i)}') 
print(torch.xpu.device_count())
print(torch.xpu.device(0))
# Pick your device index
#device_index = 1  # 0 = iGPU, 1 = Arc A770
#device = torch.device(f"xpu:{device_index}")
device = 'xpu:0'


# Use this queue with custom SYCL ops or wrappers

# Check device properties
print(torch.xpu.current_device())
print(device)

seq_len = 32

########################################
# 1. Build a Byte-Level Tokenizer/Vocab
########################################

from transformers import PreTrainedTokenizerFast

# 🔹 Change this to the actual path where your BPE tokenizer files are stored
tokenizer_path = r"C:\Users\abias\.cursor-tutor\vccdoe\mhlamodel\mhlatest-main"  

# 🔹 Load a BPE tokenizer from local files
base_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

print(f"✅ Loaded custom BPE tokenizer from: {tokenizer_path}")
print(f"📏 Vocabulary size: {base_tokenizer.vocab_size}")

# Wrap it with the hierarchical tokenizer
tokenizer = base_tokenizer


########################################
# 2. Data Extraction
########################################

def extract_data(json_data):
    """Extracts training data from JSON file and tokenizes it."""
    input_ids_list = []
    target_ids_list = []

    for item in json_data:
        conversations = item.get("conversations", [])

        if not isinstance(conversations, list) or len(conversations) < 2:
            print(f"⚠️ Skipping entry with no valid conversation: {item}")
            continue

        for i in range(len(conversations) - 1):
            user_turn = conversations[i]
            assistant_turn = conversations[i + 1]

            # Ensure we only process valid user-assistant exchanges
            if user_turn.get("from") in ["user", "human"] and assistant_turn.get("from") in ["assistant", "gpt"]:
                query = user_turn.get("value", "").strip()
                target = assistant_turn.get("value", "").strip()

                # 🔹 Ensure valid text exists before tokenizing
                if not query or not target:
                    print(f"⚠️ Skipping empty user/assistant exchange: {user_turn} -> {assistant_turn}")
                    continue  

                input_ids = tokenizer.tokenize(query)
                target_ids = tokenizer.tokenize(target)

                # 🔹 Ensure tokenized output isn't empty
                if not input_ids or not target_ids:
                    print(f"⚠️ Skipping invalid tokenized entry: {query} -> {input_ids}")
                    continue

                input_ids_list.append(input_ids)
                target_ids_list.append(target_ids)
    

    return list(zip(input_ids_list, target_ids_list))  # Ensure format is (input, target)

def load_dataset(dataset_path):

            dataset_files = os.listdir(dataset_path)
            query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(dataset_path, file)
                if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        text_data = list
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                            else:
                                data = json.load(f)
                                query_target_pairs.extend(extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]

                elif file.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['text'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'TEXT' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['TEXT'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'messages' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['messages'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        text_data.append(text)
                else:
                    print("errpr")
            if not query_target_pairs:
                print("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            text_data = []
            for query, target in query_target_pairs:
                text_data.append(f"User: {query}\nAssistant: {target}")

            logging.info(f"Loaded dataset with {len(query_target_pairs)} query/target pairs.")
            return query_target_pairs


def extract_query_target_pairs( data):
        query_target_pairs = []

        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                        query = messages[i].get("content") or messages[i].get("value", "")
                        target = messages[i + 1].get("content") or messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            elif conversation.get("text"):
                messages = conversation.get("text", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
            # Final fallback: split everything into sequence-length chunks for predictive text
            if not query_target_pairs:
                all_text = " ".join([m.get("text", "") for conversation in data for m in conversation])
                tokenized_text = tokenizer.encode(all_text, truncation=False)
                query_target_pairs = [
                    {"query": tokenized_text[i:i+seq_len], "target": tokenized_text[i:i+seq_len]}
                    for i in range(0, len(tokenized_text), seq_len)
                ]

        return query_target_pairs

def tokenize_data(query_target_pairs):

        # Select training mode
        input_ids_list = []  # Initialize for unchunked dataset
        labels_list = []  # Initialize for unchunked dataset

        for query, target in query_target_pairs:
                        input_ids, labels = _generate_training_pairs(query, target)

                        if input_ids and labels:
                            input_ids_list.append(input_ids)  # Store for training
                            labels_list.append(labels)  # Store for training
                            #print (input_ids)
                            #print(labels)
        return input_ids_list, labels_list


def _generate_training_pairs(query, target):
        # Debugging logs
        logging.debug(f"Generating Training Pairs - Query: {query}")
        logging.debug(f"Generating Training Pairs - Target: {target}")

        # Ensure inputs are valid strings before tokenization
        query_ids = tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)

        input_ids = [tokenizer.bos_token_id] + query_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]

        return input_ids, labels

def prepare_batch(input_ids, labels, seq_len):
                pad_token_id = tokenizer.pad_token_id if tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths

                #input_ids = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in input_ids]
                #labels = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in labels]

                # ✅ Compute correct padding lengths
                #input_ids = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in input_ids]
                #labels = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in labels]
                
                input_ids = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in input_ids
                ]
                logging.info("input ids torched to tensor")
                print(input_ids)
                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in labels
                ]
                logging.info("labels torched to tensor")
                print(labels)
                # Stack tensors
                input_ids = torch.stack(input_ids).to(device)
                labels = torch.stack(labels).to(device)
                data = torch.utils.data.TensorDataset(input_ids, labels)
                return data


########################################
# 3. Dataset and Collate Function
########################################

class ChatDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_seq_length):
        """Initialize dataset and tokenize the data properly."""
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # 🔹 Ensure data is correctly processed
        self.data = extract_data(json_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns exactly two elements: (input, target)."""
        return self.data[idx]

def collate_fn2(batch, max_length, tokenizer):
    src_batch, tgt_batch = zip(*batch)

    pad_token_id = tokenizer.pad_token_id or 0  # Ensure pad token is valid

    src_batch = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in src_batch]
    tgt_batch = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in tgt_batch]

    # ✅ Compute correct padding lengths
    src_batch = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in src_batch]
    tgt_batch = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in tgt_batch]
    print(src_batch)
    print(tgt_batch)
    return torch.stack(src_batch), torch.stack(tgt_batch)

def collate_fn(batch):
    """
    Collate function for standard seq2seq data. Each sample is a tuple (input_ids, target).
    Both sequences are padded/truncated to a fixed length.
    """
    input_ids = []
    labels = []
    seq_lengths = []

    if len(batch[0]) == 2:
        for query, target in batch:
            input_ids.append(query)
            labels.append(target)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return input_ids, labels
    if len(batch[0]) == 3:
        # Dataset returns: input_ids, labels, seq_lengths
        input_ids, labels, seq_lengths = zip(*batch)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        return input_ids, labels, seq_lengths

##############################################
# Positional Encoding (Standard Sin/Cos Version)
##############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Instead of erroring, simply truncate positional encodings to x.size(1)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
########################################
#Base Model
########################################


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent):
        """
        Multi-Head Latent Attention (MHLA)
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_latent: Compressed latent space dimension
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress keys/values
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct keys
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct values

    def forward(self, x, memory=None):
        """
        Forward pass with optional memory (for hierarchical tokenization)
        - x: Input tensor (batch, seq_len, d_model)
        - memory: Cached latent state (batch, d_latent) [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)  # Merge and compress
        if memory is not None:
            latent_kv = (latent_kv + memory) / 2  # Combine with previous memory

        # Reconstruct full-size keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Multi-head split
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        return output, latent_kv  # Return output and memory for next layer


class TimeAwareMultiHeadLatentAttention1(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01):
        """
        Multi-Head Latent Attention (MHLA) with Time-Aware Decay.
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_latent: Compressed latent space dimension
        - lambda_decay: Controls how quickly attention fades over time
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.lambda_decay = lambda_decay

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress keys/values
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct keys
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct values

    def forward(self, x, memory=None):
        """
        Forward pass with optional hierarchical memory.
        - x: Input tensor (batch, seq_len, d_model)
        - memory: Cached latent state (batch, d_latent) [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)  # Merge and compress
        if memory is not None:
            latent_kv = (latent_kv + memory) / 2  # Combine with previous memory

        # Reconstruct full-size keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Multi-head split
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Compute raw attention scores
        attn_scores = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / math.sqrt(self.d_model)

        # 🔹 Apply time decay to attention scores
        time_matrix = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(seq_len, seq_len)
        time_decay = torch.exp(-self.lambda_decay * torch.abs(time_matrix - time_matrix.T))  # e^(-λt)
        attn_scores = attn_scores * time_decay.unsqueeze(0).unsqueeze(0)  # Shape: (batch, heads, seq, seq)

        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        return output, latent_kv  # Return output and hierarchical memory



import time

class TimeAwareMultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01, use_wallclock_time=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.lambda_decay = lambda_decay
        self.use_wallclock_time = use_wallclock_time
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # Attention weights
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)
        self.W_mem_kv = nn.Linear(d_model, d_latent, bias=False)
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)

    def get_timestamps(self, x):
        """
        Create a timestamp tensor matching the sequence length.
        Returns: shape (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        current_time = time.time()
        # simulate per-token arrival spaced by 0.01s (or real timestamps if known)
        return torch.tensor([[current_time + i * 0.01 for i in range(seq_len)]
                             for _ in range(batch_size)], device=x.device)

    def forward(self, x, memory=None, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x)

        if memory is None:
            # Self-attention
            k = self.W_k(x)
            v = self.W_v(x)
        else:
            # Cross-attention with encoder output
            k = self.W_k(memory)
            v = self.W_v(memory)

        latent_kv = self.W_down_kv(k + v)
        # Just use memory directly:
        if memory is not None:
            latent_kv = self.W_down_kv(memory)
        else:
            latent_kv = self.W_down_kv(k + v)

        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Reshape
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Matrix multiply (batch, heads, q_seq, k_seq)
        attn_scores = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / (math.sqrt(self.d_model) + 1e-8)

        # 🔹 Apply attention mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)

        if self.use_wallclock_time:
            if memory is None:
                # Self-attention: time decay between tokens in x
                token_timestamps = self.get_timestamps(x)  # shape: (batch, seq_len)
                time_diffs = torch.abs(token_timestamps.unsqueeze(2) - token_timestamps.unsqueeze(1))  # (batch, seq, seq)
                time_decay = torch.exp(-self.lambda_decay * time_diffs)  # (batch, seq, seq)
                time_decay = time_decay.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch, heads, seq, seq)
                attn_scores = attn_scores * time_decay

            else:
                # Cross-attention: decay between decoder tokens (q) and encoder memory tokens (k)
                q_timestamps = self.get_timestamps(x)               # shape: (batch, seq_q)
                k_timestamps = self.get_timestamps(memory)          # shape: (batch, seq_k)
                time_diffs = torch.abs(q_timestamps.unsqueeze(2) - k_timestamps.unsqueeze(1))  # (batch, seq_q, seq_k)
                time_decay = torch.exp(-self.lambda_decay * time_diffs)  # (batch, seq_q, seq_k)
                time_decay = time_decay.unsqueeze(1).expand(-1, self.num_heads, -1, -1)         # (batch, heads, seq_q, seq_k)
                attn_scores = attn_scores * time_decay
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output), latent_kv



class TimeAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=seq_len, lambda_time=0.01, use_wallclock_time=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lambda_time = lambda_time
        self.use_wallclock_time = use_wallclock_time

        # 🔹 Precompute sinusoidal positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("positional_encoding", pe.unsqueeze(0), persistent=False)

    def get_timestamps(self, seq_len, batch_size, device):
        now = time.time()
        timestamps = [[now + i * 0.01 for i in range(seq_len)] for _ in range(batch_size)]
        return torch.tensor(timestamps, device=device).unsqueeze(-1)  # (batch, seq_len, 1)

    def forward(self, x, token_timestamps=None):
        """
        x: (batch, seq_len, d_model)
        token_timestamps: optional (batch, seq_len)
        """
        batch_size, seq_len, d_model = x.size()

        # 🔹 Get standard sinusoidal PE
        pos_pe = self.positional_encoding[:, :seq_len, :].to(device)
        x = x + pos_pe

        # 🔹 Add time-based information
        if self.use_wallclock_time:
            if token_timestamps is None:
                token_timestamps = self.get_timestamps(seq_len, batch_size, x.device)  # (batch, seq, 1)

            # Time signal scaled to match PE shape
            time_signal = token_timestamps * self.lambda_time  # Shape: (batch, seq_len, 1)
            time_embedding = torch.sin(time_signal * math.pi).expand(-1, -1, d_model)  # broadcast over features

            x = x + time_embedding

        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, dropout=0.1, lambda_decay=0.01):
        super().__init__()
        self.self_attn = TimeAwareMultiHeadLatentAttention(d_model, num_heads, d_latent, lambda_decay)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        ffn_out = torch.clamp(ffn_out, -5.0, 5.0)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, dropout=0.1, lambda_decay=0.01):
        super().__init__()
        self.self_attn = TimeAwareMultiHeadLatentAttention(d_model, num_heads, d_latent, lambda_decay)
        self.cross_attn = TimeAwareMultiHeadLatentAttention(d_model, num_heads, d_latent, lambda_decay)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask=None):
        attn_out, _ = self.self_attn(x, mask=mask)  # Decoder self-attention
        x = self.norm1(x + self.dropout(attn_out))

        cross_out, _ = self.cross_attn(x, memory=encoder_output)  # Cross-attention
        x = self.norm2(x + self.dropout(cross_out))

        ffn_out = self.ffn(x)
        ffn_out = torch.clamp(ffn_out, -5.0, 5.0)

        x = self.norm3(x + self.dropout(ffn_out))
        return x

class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, lambda_decay=0.01, dropout=0.1, compression_factor=4, num_frequencies=100, device=device):
        super().__init__()
        self.embed_size = embedding_dim
        self.d_latent = embedding_dim // compression_factor
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.device=device

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = TimeAwarePositionalEncoding(embedding_dim, max_len=seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(embedding_dim, num_heads, self.d_latent, dropout, lambda_decay)
            for _ in range(num_layers)
        ])


        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embedding_dim, num_heads, self.d_latent, dropout, lambda_decay)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src_ids, tgt_ids):
        # --- INPUT EMBEDDING ---
        src_ids.to(device)
        tgt_ids.to(device)
        self.to(device)
        src_emb = self.token_embedding(src_ids)
        src_emb = self.pos_encoder(src_emb)
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)

        # --- DECODER PROCESS ---
        tgt_emb = self.token_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, src_emb)

        return self.fc_out(tgt_emb)
    


# === Hypertransformer ===
class HyperTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads, weight_dim):
        super().__init__()
        self.embed = nn.Linear(input_dim, model_dim)
        encoder = nn.TransformerEncoderLayer(model_dim, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.out = nn.Linear(model_dim, weight_dim)

    def forward(self, x):
        x.to(device)
        x = self.embed(x.float().to(device)).to(device)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.out(pooled)

class DualWeightLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_fwd = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.W_bwd = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, direction="forward"):
        if direction == "forward":
            return F.linear(x, self.W_fwd, self.bias)
        elif direction == "backward":
            return F.linear(x, self.W_bwd.t())  # interpret as error or influence backward
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

class DualWeightMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = DualWeightLayer(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.layer2 = DualWeightLayer(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.act(self.layer1(x, direction="forward")), direction="forward")

    def backward_pass(self, loss_grad):
        """
        Simulated backward dynamics using backward weights.
        Could be an activation map or error signal flowing in reverse.
        """
        x = self.layer2(loss_grad, direction="backward")
        x = self.act(x)
        x = self.layer1(x, direction="backward")
        return x
    
class DualWeightLayerWithPlasticity(DualWeightLayer):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.plasticity = nn.Parameter(torch.zeros(out_dim, in_dim))

    def forward(self, x, direction="forward", apply_plasticity=False):
        W = self.W_fwd
        if apply_plasticity:
            W = W + self.plasticity * torch.bmm(x.unsqueeze(2), x.unsqueeze(1)).mean(0)
        return F.linear(x, W, self.bias)


class DualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Forward attention
        self.qkv_proj_fwd = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_out_proj_fwd = nn.Linear(embed_dim, embed_dim)

        # Backward attention
        self.qkv_proj_bwd = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_out_proj_bwd = nn.Linear(embed_dim, embed_dim)

        # Shared normalization and FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def attention(self, x, qkv_proj, out_proj):
        B, T, C = x.size()
        qkv = qkv_proj(x)  # shape: (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head attention
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return out_proj(attn_output)

    def forward(self, x, direction="forward"):
        residual = x

        if direction == "forward":
            x = self.attention(x, self.qkv_proj_fwd, self.attn_out_proj_fwd)
        elif direction == "backward":
            x = self.attention(x, self.qkv_proj_bwd, self.attn_out_proj_bwd)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

        x = self.norm1(x + residual)
        x = self.norm2(x + self.ff(x))
        return x

class DualTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_layers=2, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            DualAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x, direction="forward"):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, direction=direction)
        return self.out(x)


class PlasticDualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, plasticity=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.plasticity_enabled = plasticity

        # Attention projections
        self.qkv_proj_fwd = nn.Linear(embed_dim, 3 * embed_dim)
        self.qkv_proj_bwd = nn.Linear(embed_dim, 3 * embed_dim)

        self.out_proj_fwd = nn.Linear(embed_dim, embed_dim)
        self.out_proj_bwd = nn.Linear(embed_dim, embed_dim)

        # Plasticity gate (learnable gating function)
        self.plasticity_gate_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def multihead_attention(self, x, qkv_proj, out_proj):
        B, T, C = x.shape
        H = self.num_heads
        D = C // H

        qkv = qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / D**0.5
        attn_weights = F.softmax(attn, dim=-1)
        attn_out = attn_weights @ v  # (B, H, T, D)

        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return out_proj(out), attn_out  # return attention output for update use

    def forward(self, x, direction="forward"):
        residual = x
        B, T, C = x.shape

        if direction == "forward":
            attn_out, fwd_heads = self.multihead_attention(x, self.qkv_proj_fwd, self.out_proj_fwd)
            x = self.norm1(attn_out + residual)
            x = self.norm2(self.ff(x) + x)
            return x  # standard output

        elif direction == "backward" and self.plasticity_enabled:
            attn_out, bwd_heads = self.multihead_attention(x, self.qkv_proj_bwd, self.out_proj_bwd)

            # compute plasticity gate signal
            gate_input = x.mean(dim=1)  # (B, C)
            gate = torch.sigmoid(self.plasticity_gate_proj(gate_input))  # (B, 1)
            print(f"Plasticity Gate (mean across batch): {gate.mean().item():.4f}")

            # Hebbian-like update: outer product between attention heads from fwd and bwd
            # Simulating: ΔW = η * x_fwdᵀ · x_bwd
            if hasattr(self, "last_fwd_heads"):
                hebb_update = torch.einsum('bhte,bhue->bhtu', self.last_fwd_heads, bwd_heads).mean(0)  # (H, T, T, U)
                # Project to match forward weight shape
                update_tensor = hebb_update.mean(dim=(1,2))  # crude average per-head, shape (H, U)
                update_tensor = update_tensor.view_as(self.qkv_proj_fwd.weight[:self.embed_dim])  # align with W_q
                self.qkv_proj_fwd.weight.data[:self.embed_dim] += 0.001 * gate.mean() * update_tensor

            self.last_bwd_heads = bwd_heads.detach()
            return x  # backward signal output

        elif direction == "backward":
            # If not updating, just return backward attention output
            attn_out, _ = self.multihead_attention(x, self.qkv_proj_bwd, self.out_proj_bwd)
            return attn_out

        else:
            raise ValueError("Unknown direction.")

class PlasticDualTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_layers=2, num_heads=4, plasticity=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            PlasticDualAttentionBlock(embed_dim, num_heads, plasticity=plasticity)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x, direction="forward"):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, direction=direction)
        return self.out(x)

########################################
# 5. Training Loop
########################################


# === Utilities ===
def flatten_model_weights(model):
    return torch.cat([p.data.flatten() for p in model.parameters() if p.requires_grad])


def apply_offset_attn_delta(model, base_weights, delta, batch_idx, max_update=8192, kernel_size=5):
    """
    Vectorized: Applies a soft-local delta around a sliding offset.
    """
    flat_dim = base_weights.numel()
    updated = base_weights.clone()

    # Define offset
    offset = (batch_idx * max_update) % flat_dim
    center_start = offset
    center_end = min(offset + max_update, flat_dim)

    # Clip delta size to match available range
    clipped_delta = delta[:center_end - center_start]

    # Apply primary delta
    updated[center_start:center_end] += clipped_delta

    # Optional: smooth attenuated outer window
    for i in range(1, kernel_size + 1):
        decay = 1.0 / math.sqrt(i + 1)

        left_idx = max(0, center_start - i)
        right_idx = min(flat_dim, center_end + i)

        if left_idx < center_start:
            updated[left_idx:center_start] += decay * clipped_delta[:center_start - left_idx]
        if center_end < right_idx:
            updated[center_end:right_idx] += decay * clipped_delta[:right_idx - center_end]

    # Write back
    #load_flat_weights(model, updated)
    print("delta shape:", delta.shape)




def load_flat_weights(model, flat_vector):
    offset = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        numel = param.numel()
        param.data.copy_(flat_vector[offset:offset+numel].view_as(param))
        offset += numel

def force_move_buffers(model, device):
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            setattr(model, name, buffer.to(device))

def summarize_attention(model):
    fwd_summary = []
    bwd_summary = []
    for block in model.blocks:
        if hasattr(block, 'last_fwd_heads') and block.last_fwd_heads is not None:
            fwd_summary.append(block.last_fwd_heads.mean().unsqueeze(0))
        if hasattr(block, 'last_bwd_heads') and block.last_bwd_heads is not None:
            bwd_summary.append(block.last_bwd_heads.mean().unsqueeze(0))
    return torch.cat(fwd_summary + bwd_summary, dim=0).view(1, -1)

# Then inside training loop:
#summary_input = summarize_attention(model)
#flat_input = summary_input.detach()

# === Optimizer Comparison and Training ===
def evaluate_hyper_vs_adam(hyper_model, base_model, batch, val_batch, loss_fn, lr=0.001):
    orig_weights = flatten_model_weights(base_model).detach().to(device)

    input_batch, target_batch = batch
    
    base_model.eval()
    base_model.zero_grad()
    base_model.to(device)
    pred = base_model(input_batch.to(device), target_batch.to(device))
    loss = loss_fn(pred.view(-1, pred.size(-1)), target_batch.view(-1))

    loss.backward()

    adam_update = torch.cat([
        (-lr * p.grad.view(-1)) if p.grad is not None else torch.zeros_like(p.view(-1))
        for p in base_model.parameters()
    ])
    weights_adam = orig_weights + adam_update

    hyper_model.to(device)
    flat_input = torch.cat([input_batch.flatten(1), target_batch.flatten(1)], dim=1).unsqueeze(0).to(device)
    delta_hyper = hyper_model(flat_input.to(device)).squeeze(0).to(device)
    weights_hyper = orig_weights + delta_hyper

    input_val, target_val = val_batch
    val_decoder_input, target_val = prepare_decoder_input_and_target(target_val)

    base_model.eval()
    load_flat_weights(base_model, weights_adam)
    loss_adam = loss_fn(base_model(input_val, val_decoder_input).view(-1, base_model.fc_out.out_features), target_val.view(-1)).item()

    load_flat_weights(base_model, weights_hyper)
    loss_hyper = loss_fn(base_model(input_val, val_decoder_input).view(-1, base_model.fc_out.out_features), target_val.view(-1)).item()

    if loss_hyper < loss_adam:
        load_flat_weights(base_model, weights_hyper)
        return loss_hyper, 'hyper'
    else:
        load_flat_weights(base_model, weights_adam)
        return loss_adam, 'adam'
    

def prepare_decoder_input_and_target(target):
    """
    Prepares inputs and targets for teacher forcing when <BOS> is auto-generated by the tokenizer.
    - target: Tensor of shape (batch_size, seq_len)
    Returns:
    - decoder_input: Shifted target, including <BOS>
    - target_output: Original target
    """
    # Shift target to the right to form the decoder input
    decoder_input = torch.zeros_like(target)
    decoder_input[:, 1:] = target[:, :-1]  # Shift right
    decoder_input[:, 0] = target[:, 0]     # Copy the <BOS> from the target

    # The output is the target sequence itself (including <EOS>)
    target_output = target
    
    return decoder_input, target_output


def build_custom_validation_batch(tokenizer, seq_len=seq_len, device=device):
    query_strings = [
        "1. What is 17 + 35?",
        "2. Solve for x: 2x + 5 = 13",
        "3. What is the derivative of x^2?",
        "4. What is the integral of x dx?",
        "5. What is the plural of 'analysis'?",
        "6. Is this sentence correct? 'He go to school every day.'",
        "7. What is the first law of Robotics?",
        "8. What is the secpnd law of robotics?",
        "9. What is the third law of robotics?,",
        "10. What is the zeroth law of robotics?",
        "11. What does this Python function return? def square(x): return x * x",
        "12. Write a function in Python that checks if a number is prime.",
        "13. What is the derivative of a function x^3 according to calculus?",
        "14. Describe the integral of a function x^3 according to calculus, please."
    ]

    target_strings = [
        "1. 52",
        "2. x = 4",
        "3. 2x",
        "4. (1/2)x^2 + C",
        "5. analyses",
        "6. No, it should be: 'He goes to school every day.'",
        "7. 1. A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
        "8. 2. A robot must obey orders given by humans except where such orders would conflict with the First Law.",
        "9. 3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
        "10. 0. A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",
        "11. It returns the square of x.",
        "12. def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "13. The derivative of x^3 by the power law for derivatives would be 3x^2.",
        "14. According to the integral power law the integral of x^3 would be (x^2)/2."
    ]

    input_ids, target_ids = [], []
    for query, target in zip(query_strings, target_strings):
        q_ids = tokenizer.encode(query, max_length=seq_len, truncation=True, padding='max_length')
        a_ids = tokenizer.encode(target, max_length=seq_len, truncation=True, padding='max_length')

        input_ids.append(q_ids)
        target_ids.append(a_ids)

    return input_ids, target_ids

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for batch_idx, (src, target) in enumerate(dataloader):
        
        src = src.to(device)
        target = target.to(device)
        decoder_input, target_labels = prepare_decoder_input_and_target(target)
        decoder_input = decoder_input.to(device)
        target_labels = target_labels.to(device)
        optimizer.zero_grad()
        model = model.to(device)
        # 🔹 Get predictions & rule-modified embeddings
        output = model(src, decoder_input).to(device)
        #output = model(src, target_labels)
        # 🔹 Ensure `output` and `target_labels` have the same sequence length
        seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
        output = output[:, :seq_len, :]  # Truncate logits if too long
        target_labels = target_labels[:, :seq_len]  # Truncate targets if too long

        # 🔹 Flatten for cross_entropy()
        loss = criterion(output.reshape(-1, output.shape[-1]), target_labels.reshape(-1))
        n+=1
        print(f"Iteration {n}, Loss: {loss.item()}")
        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 Warning: NaN or Inf detected in loss! Skipping update.")
            return

        loss.backward()

        # 🔹 Track how rules affected loss
        prev_loss = loss.item()
        # Clip gradients to prevent exploding values
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 🔹 After updating, re-run forward to see new loss
        output_new = model(src, decoder_input)
        #output_new = model(src)
        new_loss = criterion(output_new[:, :seq_len, :].reshape(-1, output_new.shape[-1]), 
                                 target_labels.reshape(-1)).item()
        #Test rules and generate new ones                          
        loss_diff = prev_loss - new_loss  # Negative means rule improved loss
            #Test rules and generate new ones                          
        model.rule_transform.update_rule_scores(src.to(device), loss_diff)

        val_input, val_target = build_custom_validation_batch(tokenizer, device=device)
        val_dec_input, val_target = prepare_decoder_input_and_target(val_target)

        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_step(hyper, base_model, data_batch, targets, loss_fn):
    base_model.eval()
    n=0
    # Flatten input + target per sample as token for transformer
    B = data_batch.shape[0]
    flat_input = torch.cat([data_batch, targets], dim=1).unsqueeze(0).to(device)  # (1, B, in+out)

    # Get original weights
    original_weights = base_model.get_weights().detach().to(device)
    hyper.to(device)
    delta_weights = hyper(flat_input.to(device)).to(device)
    new_weights = original_weights + delta_weights[0]
    base_model.set_weights(new_weights)

    decoder_input, targets = prepare_decoder_input_and_target(targets)

    # Evaluate loss with new weights
    preds = base_model(data_batch, decoder_input)
    loss = loss_fn(preds.view(-1, preds.size(-1)), targets.view(-1))
    n+=1
    print(f"Iteration {n}, Loss: {loss.item()}")


    return loss

def train_hyper(hyper, model, dataloader, val_loader, opt, loss_fn, mode, device=device):
    model.train()
    total_loss = 0
    n = 0
    for batch_idx, (src, target) in enumerate(dataloader):
        if mode == 'eval':
            decoder_input, target = prepare_decoder_input_and_target(target)

            vb, vy = next(iter(val_loader))
            loss, winner = evaluate_hyper_vs_adam(hyper, model, (src, target), (vb, vy), loss_fn)
            if winner == 'adam':
                opt.zero_grad()
                pred = hyper(torch.cat([src.flatten(1), target.flatten(1)], dim=1).unsqueeze(0))
                pred_loss = F.mse_loss(pred, flatten_model_weights(model).unsqueeze(0))
                pred_loss.backward()
                opt.step()
            print(f"Step {batch_idx}: Loss {loss:.4f}, Winner: {winner}")
        src = src.to(device)
        target = target.to(device)

        loss = train_step(hyper, model, src, target, loss_fn)
        
        n+=1
        print(f"Iteration {n}, Loss: {loss.item()}")


        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_hyperplastic(model, hyper, dataloader, val_loader, loss_fn, hyper_opt, device):
    model.train()
    hyper.train()
    total_loss = 0.0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)

        # Save weights
        w_before = flatten_model_weights(model).detach()

        # Prepare input for hypertransformer
        with torch.no_grad():
            src_embed = model.embed(src).mean(dim=1)  # (B, C)
            tgt_embed = model.embed(tgt).mean(dim=1)  # (B, C)

        flat_input = torch.cat([src_embed, tgt_embed], dim=1).unsqueeze(0).float()  # (1, 2C)

        #memory blocks for later testing
        # Assume model.blocks stores activations
        #forward_summary = []
        #backward_summary = []

        #for block in model.blocks:
        #    if hasattr(block, 'last_fwd_heads'):
        #        forward_summary.append(block.last_fwd_heads.mean().unsqueeze(0))
         #   if hasattr(block, 'last_bwd_heads'):
          #      backward_summary.append(block.last_bwd_heads.mean().unsqueeze(0))

        #if forward_summary and backward_summary:
         #   summary = torch.cat(forward_summary + backward_summary, dim=0)
          #  flat_input = summary.view(1, -1)
          
        #####OR difference of means
        #with torch.no_grad():
         #   src_embed = model.embed(src).mean(dim=1)
          #  tgt_embed = model.embed(tgt).mean(dim=1)

        # Input vector = [src, tgt, diff]
        #flat_input = torch.cat([src_embed, tgt_embed, tgt_embed - src_embed], dim=1).unsqueeze(0)

        # Predict delta weights
        delta = hyper(flat_input.to(device)).squeeze(0)
        ##oldload_flat_weights(model, w_before + delta)
        apply_offset_attn_delta(model, w_before, delta, batch_idx, max_update=8192, kernel_size=5)

        # Forward pass with updated model
        preds = model(src, direction="forward")
        loss = loss_fn(preds.view(-1, preds.size(-1)), tgt.view(-1))

        # Validation pass to assess performance
        with torch.no_grad():
            try:
                val_src, val_tgt = next(val_loader_iter)
            except:
                val_loader_iter = iter(val_loader)
                val_src, val_tgt = next(val_loader_iter)
            val_src, val_tgt = val_src.to(device), val_tgt.to(device)
            val_preds = model(val_src, direction="forward")
            val_loss = loss_fn(val_preds.view(-1, val_preds.size(-1)), val_tgt.view(-1)).item()

        # Train hypertransformer to reduce val loss
        hyper_opt.zero_grad()
        loss.backward()
        hyper_opt.step()

        # Backward plasticity for W_fwd update
        with torch.no_grad():
            _ = model(tgt, direction="backward")

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_plastic(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    model.to(device)
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)

        # === Forward pass ===
        preds = model(src, direction="forward")
        loss = loss_fn(preds.view(-1, preds.size(-1)), tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === Backward plasticity signal ===
        with torch.no_grad():
            _ = model(tgt, direction="backward")  # simulate backward influence

        total_loss += loss.item()
        n+=1
        print(f"Iteration {n}, Loss: {loss.item()}")


    return total_loss / len(dataloader)

########################################
#6. inference
########################################


# Inference function for autoregressive decoding.
def inference(model, input_text, max_seq_length, device, max_generated=30):
                    model.eval()
                    with torch.no_grad():
                        # Tokenize the prompt and move to the correct device.
                        input_ids = base_tokenizer.tokenize(input_text)
                        input_ids = base_tokenizer.encode(input_text)
                        print(input_ids)
                        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
                        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

                        # ✅ Compute correct padding lengths
                        #input_ids = [torch.cat([seq, torch.full((max(0, max_generated - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in input_ids]
                        # Pad input_ids to the maximum sequence length
                        generated_text = input_ids
                        generated = []
                        logging.debug(f"Padded input_ids Shape: {input_ids.shape}")
                        print(input_ids.shape)

                        # Choose a start token for the dummy target.
                        # Here we use tokenizer.eos_token_id if available; otherwise, fallback to tokenizer.pad_token_id.
                        bos_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
                        eos_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                        eos_token  = torch.tensor([[eos_token]], device=device)

                        tgt_ids = torch.tensor([[bos_token]], device=device)
                        print(tgt_ids.shape)
                        tgt_ids = torch.cat([tgt_ids, input_ids], dim=1)
                        logging.info(f"tgt_ids: {tgt_ids}")

                        # Keep track of the original input length
                        input_length = input_ids.size(1)

                        for _ in range(seq_len - input_ids.size(1)):
                            # Generate the target mask for the current target sequence length.
                            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
                            # Forward pass through the model
                            #outputs = model(input_ids, tgt_ids)
                            outputs = model(input_ids)
                            logging.debug(f"output shape: {outputs.shape}")

                            # Get logits for the last token and apply argmax to get the next token ID
                            next_token_logits = outputs[:, -1, :]  # Get the logits for the last position
                            repetition_penalty = 1.2  # Adjust for stronger penalty
                            # Apply repetition penalty while excluding special tokens like PAD (0)
                            for token in set(generated_text[0].tolist()):
                                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                                    next_token_logits[0, token] /= repetition_penalty


                            top_p = 0.9  # Cumulative probability threshold
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            filtered_logits = next_token_logits.clone()
                            filtered_logits[sorted_indices_to_remove] = float('-inf')

                            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                            logging.debug(f"next_token_logits: {next_token_logits}")
                            logging.debug(f"next_token_logits shape: {next_token_logits.shape}")
                            logging.debug(f"next_token_id shape: {next_token_id.shape}")
                            logging.debug(f"next_token_id: {next_token_id}")
                            # Append the new token to the target sequence.
                            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
                            logging.debug(f"tgt_ids: {tgt_ids}")
                            input_ids = input_ids[input_ids != tokenizer.pad_token_id].unsqueeze(0)
                            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                            logging.debug(f"input_ids: {input_ids}")
                            generated.append(tokenizer.decode(next_token_id[0].tolist()))
                            logging.debug(f"generated_text: {generated_text}")
                            #print(tgt_ids)
                            # Stop generation if eos_token is generated
                            if next_token_id.item() == eos_token or tgt_ids.size(1) >= max_seq_length:
                                break

                    return generated


def generate(model, input_text, max_seq_length, device, chunk_size=30, max_generated=120):
    """
    Generates text using autoregressive decoding, ensuring chunk alignment.
    """

    model.eval()
    with torch.no_grad():
        # 🔹 Tokenize input and move to the correct device
        input_ids = base_tokenizer.encode(input_text, return_tensors="pt").to(device)
        print(f"🔍 Initial input tokens: {input_ids.tolist()}")

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_token_id

        # 🔹 Ensure input length aligns with chunk size
        original_length = input_ids.shape[1]
        pad_length = (chunk_size - (original_length % chunk_size)) % chunk_size
        if pad_length > 0:
            pad_tokens = torch.full((1, pad_length), pad_token_id, dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, pad_tokens], dim=1)

        print(f"✅ Padded input size: {input_ids.shape}")

        # 🔹 Initialize output storage
        generated_tokens = input_ids.clone().tolist()[0]  # Convert to Python list

        # 🔹 Autoregressive decoding loop
        for _ in range(max_generated):
            with torch.no_grad():
                output = model(input_ids)  # Forward pass
            
            print(f"🔹 Model Output Shape: {output.shape}")  # Debug Output

            # 🔹 Ensure output shape is correct before applying `argmax`
            if output.shape[-1] != tokenizer.vocab_size:
                print(f"⚠️ Warning: Output vocab size mismatch! Expected {tokenizer.vocab_size}, got {output.shape[-1]}")
                break  # Prevent invalid indexing
            
            # 🔹 Select next token (greedy decoding)
            next_token = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)

            # 🔹 Convert tensor to integer
            next_token_id = next_token.item()
            generated_tokens.append(next_token_id)

            # 🔹 Stop if EOS token is reached
            if next_token_id == eos_token_id:
                print(f"🛑 Generation stopped: EOS token reached.")
                break

            # 🔹 Append new token and **REMOVE FIRST TOKEN** to maintain sequence history
            input_ids = torch.cat([input_ids[:, 1:], next_token], dim=1)

            print(f"🔹 New token: {next_token_id}, Updated input size: {input_ids.shape}")

        # 🔹 Decode final output
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text


def load_json_file(file_path):
    """Load the JSON dataset file properly."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # 🔹 Ensure it's properly parsed
            if not isinstance(data, list):
                raise ValueError("🚨 Loaded data is not a list of dictionaries.")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"🚨 Failed to parse JSON: {e}")

def generate_2(model, prompt, tokenizer, seq_len, device, max_generated=120, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    generated_tokens = []
    model.to(device)
    with torch.no_grad():
        # Tokenize prompt → fixed encoder input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        encoder_input_len = input_ids.size(1)

        # Pad encoder input to max model length
        if encoder_input_len < seq_len:
            pad_len = seq_len - encoder_input_len
            pad_token_id = tokenizer.pad_token_id or 0
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :seq_len]

        # Encoder is static throughout generation
        encoder_input_ids = input_ids.to(device)

        # Setup initial decoder input
        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        tgt_ids = torch.tensor([[bos_token_id]], device=device)

        for _ in range(max_generated):
            # Forward pass through model
            encoder_input_ids.to(device)
            tgt_ids.to(device)
            model.to(device)
            outputs = model(encoder_input_ids, tgt_ids).to(device)
            logits = outputs[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            for token in set(tgt_ids[0].tolist()):
                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                    logits[0, token] /= repetition_penalty

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            filtered_logits = logits.clone()
            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # Stop at EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append and continue
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())

            # Pad if needed to align with model
            if tgt_ids.size(1) > seq_len:
                tgt_ids = tgt_ids[:, -seq_len:]

    return tokenizer.decode(generated_tokens)

def generate3(model, prompt, steps, tokenizer, device):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = input_ids.clone()

    for _ in range(steps):
        input_crop = output[:, -model.seq_len:]  # crop to context length
        logits = model(input_crop, direction="forward")
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        output = torch.cat([output, next_token], dim=1)

    return tokenizer.decode(output[0])


########################################
# 7. Main Function
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r"C:\Users\abias\.cursor-tutor\vccdoe\mhlamodel\mhlatest-main\data", help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=14, help='Batch size for training')
    parser.add_argument('--max_seq_length', type=int, default=seq_len, help='Fixed maximum sequence length')
    parser.add_argument('--optim_mode', type=str, choices=['adam', 'hyper', 'eval'], default='adam')
    parser.add_argument('--train_mode', type=str, default='plastic', choices=['plastic', 'hyperplastic'])

    args = parser.parse_args()
    
    device = 'xpu:0'
    

    # ***** NEW: Load tokenizer from file instead of building from the data *****

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    # Load dataset correctly
    #json_data = load_json_file(args.data)

    # Pass parsed JSON instead of raw file path
    data = load_dataset(args.data)
    inputs, targets = tokenize_data(data)
    dataset = prepare_batch(inputs, targets, args.max_seq_length)
    val_inputs, val_targets = build_custom_validation_batch(tokenizer)
    val_dataset = prepare_batch(val_inputs, val_targets, args.max_seq_length)
    # 🔹 Ensure dataset isn't empty
    if len(dataset) == 0:
        raise ValueError("🚨 Dataset is empty after filtering invalid entries! Check your dataset.")

    # Use a lambda to pass the fixed length to collate_fn.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch))
    seq_length = args.max_seq_length
    d_model = 32
    n_layers = 2
    n_heads = 2
    mode = args.optim_mode

    torch.set_printoptions(sci_mode=False)
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()  # if using CUDA

    # Replace model initialization
    model = PlasticDualTransformer(vocab_size, d_model, seq_len, n_layers, n_heads).to(device)
    #model = Transformer_Model(vocab_size, d_model, n_layers, n_heads, seq_len).to(device)

   #model = Transformer_Model(vocab_size, d_model, n_layers, n_heads, seq_len).to(device)
    hyper_model = HyperTransformer(input_dim=seq_len*2, model_dim=d_model, num_layers=2, num_heads=n_heads, weight_dim=8192).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5)
    #model, optimizer = ipex.optimize(model.to(device), optimizer=optimizer, dtype=torch.bfloat16).to(device)

    for buffer_name, buffer in model.named_buffers():
        setattr(model, buffer_name, buffer.to(device))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(1, args.epochs + 1):
        #avg_loss = train_model(model, dataloader, optimizer, criterion, device)
        #avg_loss = train_hyper(hyper_model, model, dataloader, val_loader, optimizer, criterion, mode)
        #avg_loss = train_plastic(model, dataloader, optimizer, criterion, device)
        avg_loss = train_hyperplastic(model, hyper_model, dataloader, val_loader, criterion, optimizer, device)

        #if args.train_mode == 'hyperplastic':
            #loss = train_hyperplastic(model, hyper_model, dataloader, val_loader, criterion, hyper_model, device)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
    
    
    # Set the model to evaluation mode and perform inference.
    prompt = "What is the critical temperature of a superconducting thin film made of lead with a thickness of 100 nm?"
    #generated_text = hierarchical_inference(model, prompt, seq_length, device)
    #generated_text = inference(model,prompt, seq_length, device)
    #generated_text = generate_2(model,prompt, base_tokenizer, seq_length, device)
    generated_text = generate3(model, prompt, seq_len, base_tokenizer, device)

    print("Generated text:")
    print(generated_text)

if __name__ == '__main__':
    main()




