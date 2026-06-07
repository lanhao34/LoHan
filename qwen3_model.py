import math
from dataclasses import dataclass
from typing import Unpack
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from new_checkpoint import checkpoint as new_checkpoint
from checkpoint import save_on_cpu, get_selective_offloading_checkpoint_modes2
import json
from collections import namedtuple
from nvme_ds.partitionrd_activation_swapper import AsyncPartitionedActivationSwapper
# from flash_attn import flash_attn_qkvpacked_func
from transformers import Qwen3Model

act_stream = torch.cuda.Stream()
chp_id = [0]
chp_list = []
act_swapper = None
is_swap_and_recompute = 0

def set_training(args):
    global act_stream, chp_id, chp_list, act_swapper, is_swap_and_recompute
    packed = torch.empty(
            (args.batch_size, args.max_seq_len, args.hidden_dim),
            dtype=torch.float16,
            pin_memory=True)
    for i in range(2 * args.num_layers):
        # packed = torch.ones(
        #         1,
        #         dtype=torch.float16,
        #         pin_memory=True)
        chp_list.append(packed)

    def json_object_hook(d): 
        return namedtuple('X', d.keys())(*d.values())
    with open(args.sb_config) as f: 
        ds_config = json.load(f, object_hook=json_object_hook)

    # act_swapper = AsyncPartitionedActivationSwapper(ds_config, torch.float16)
    act_swapper = None
    is_swap_and_recompute = args.is_swap_and_recompute

def new_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
    def func1(x):
        residual = x
        x = self.input_layernorm(x)
        # Self Attention
        x, _ = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        x = residual + x
        return x

    def func2(x):
        # Fully Connected
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x
    with save_on_cpu(pin_memory=True, act_stream=act_stream, chp_id = chp_id, chp_list = chp_list, act_swapper=act_swapper):
        hidden_states = checkpoint(func1, hidden_states)
    with save_on_cpu(pin_memory=True, act_stream=act_stream, chp_id = chp_id, chp_list = chp_list, act_swapper=act_swapper):
        hidden_states = checkpoint(func2, hidden_states)
    
    return hidden_states

def get_qwen3_model(config):
    config.use_cache = False
    torch.set_default_dtype(torch.float16)
    torch.set_default_device('cpu')

    model = Qwen3Model(config=config)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    model.token_emb = model.embed_tokens
    model.pos_emb = model.rotary_emb
    for layer in model.layers:
        layer.forward = new_forward.__get__(layer, layer.__class__)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    return model
