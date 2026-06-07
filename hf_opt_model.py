import json
from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import OPTModel

from checkpoint import save_on_cpu


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
        pin_memory=True,
    )
    for _ in range(2 * args.num_layers):
        chp_list.append(packed)

    def json_object_hook(d):
        return namedtuple("X", d.keys())(*d.values())

    with open(args.sb_config) as f:
        json.load(f, object_hook=json_object_hook)

    act_swapper = None
    is_swap_and_recompute = args.is_swap_and_recompute


def new_forward(
    self,
    hidden_states,
    attention_mask=None,
    layer_head_mask=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,
):
    if output_attentions or use_cache:
        raise NotImplementedError("Ratel HF OPT baseline disables attentions/cache during training.")

    def func(x):
        output = self._ratel_original_forward(
            x,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        )
        return output[0]

    with save_on_cpu(pin_memory=True, act_stream=act_stream, chp_id=chp_id, chp_list=chp_list, act_swapper=act_swapper):
        hidden_states = checkpoint(func, hidden_states)

    return (hidden_states,)


def get_hf_opt_model(config):
    config.use_cache = False
    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cpu")

    model = OPTModel(config=config)
    lm_head_dim = getattr(config, "word_embed_proj_dim", config.hidden_size)
    model.lm_head = nn.Linear(lm_head_dim, config.vocab_size, bias=False)
    model.token_emb = model.decoder.embed_tokens
    model.pos_emb = model.decoder.embed_positions
    if getattr(config, "tie_word_embeddings", False) and model.lm_head.weight.shape == model.decoder.embed_tokens.weight.shape:
        model.lm_head.weight = model.decoder.embed_tokens.weight
    for layer in model.decoder.layers:
        layer._ratel_original_forward = layer.forward
        layer.forward = new_forward.__get__(layer, layer.__class__)

    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    return model
