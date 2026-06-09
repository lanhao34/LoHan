from ratel_init import Init, SB_hook
from ratel_optimizer import SB_optimizer
from see_mem import see_memory_usage
from nvtx import nvtx_wrap
import argparse
import json
import os
import tempfile
import torch
import torch.nn as nn
from op_ds.ops.CPUAdam import DeepSpeedCPUAdam
import torch.multiprocessing as mp
from time import time
from utils import priority_sort, get_act_swap_list
from nvme_ds.utils import print_object
from ratel_stats import phase, print_summary, record_step, reset_cuda_peaks


def _configure_model(args):
    if args.model_name:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        model_name_lower = args.model_name.lower()
        config_model_type = getattr(config, "model_type", "").lower()
        if "qwen3_moe" in config_model_type or "qwen3-moe" in model_name_lower:
            from qwen3_model_moe import act_stream, get_qwen3_moe_model, set_training
            args.hidden_dim = config.hidden_size
            args.num_heads = config.num_attention_heads
            args.num_layers = config.num_hidden_layers
            args.vocab_size = config.vocab_size
            return "qwen3_moe", config, get_qwen3_moe_model, act_stream, set_training
        if "qwen" in config_model_type or "qwen" in model_name_lower:
            from qwen3_model import act_stream, get_qwen3_model, set_training
            args.hidden_dim = config.hidden_size
            args.num_heads = config.num_attention_heads
            args.num_layers = config.num_hidden_layers
            args.vocab_size = config.vocab_size
            return "qwen3", config, get_qwen3_model, act_stream, set_training
        if config_model_type == "opt" or "opt" in model_name_lower:
            from hf_opt_model import act_stream, get_hf_opt_model, set_training
            args.hidden_dim = config.hidden_size
            args.num_heads = config.num_attention_heads
            args.num_layers = config.num_hidden_layers
            args.vocab_size = config.vocab_size
            return "hf_opt", config, get_hf_opt_model, act_stream, set_training
        raise ValueError(f"Unsupported HF model for Ratel baseline: {args.model_name} (model_type={config_model_type})")

    from gpt_model import GPT2Model, GPT2Config, act_stream, set_training

    if args.model_size == '1.3B':
        args.hidden_dim = 2048
        args.num_heads = 32
        args.num_layers = 24
    elif args.model_size == '3B':
        args.hidden_dim = 2560
        args.num_heads = 32
        args.num_layers = 32
    elif args.model_size == '7B':
        args.hidden_dim = 4096
        args.num_heads = 32
        args.num_layers = 32
    elif args.model_size == '13B':
        args.hidden_dim = 5120
        args.num_heads = 40
        args.num_layers = 40
    elif args.model_size == '30B':
        args.hidden_dim = 7168
        args.num_heads = 56
        args.num_layers = 48
    elif args.model_size == '66B':
        args.hidden_dim = 9216
        args.num_heads = 72
        args.num_layers = 64

    assert args.hidden_dim % args.num_heads == 0
    args.dim_head = args.hidden_dim // args.num_heads
    config = GPT2Config(
        dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_head=args.dim_head,
        max_seq_len=args.max_seq_len,
        attn_pdrop=0.1,
        dropout=0.1,
        vocab_size=args.vocab_size,
        layer_norm_epsilon=1e-5,
    )
    return "gpt", config, lambda cfg: GPT2Model(cfg).half(), act_stream, set_training

def _set_nested(config, path, value):
    if value is None:
        return
    target = config
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _prepare_runtime_config(args):
    with open(args.sb_config) as f:
        config = json.load(f)

    _set_nested(config, ["zero_config", "offload_param", "nvme_path"], args.nvme_path)
    _set_nested(config, ["zero_config", "offload_optimizer", "nvme_path"], args.nvme_path)
    _set_nested(config, ["zero_config", "offload_act", "nvme_path"], args.nvme_path)
    _set_nested(config, ["zero_config", "offload_param", "buffer_count"], args.param_buffer_count)
    _set_nested(config, ["zero_config", "offload_param", "buffer_size"], args.param_buffer_size)
    _set_nested(config, ["zero_config", "offload_param", "max_in_cpu"], args.param_max_in_cpu)
    _set_nested(config, ["zero_config", "offload_optimizer", "buffer_count"], args.optimizer_buffer_count)
    _set_nested(config, ["zero_config", "offload_act", "buffer_count"], args.activation_buffer_count)
    _set_nested(config, ["zero_config", "offload_act", "buffer_size"], args.activation_buffer_size)
    _set_nested(config, ["zero_config", "offload_act", "max_in_cpu"], args.activation_max_in_cpu)
    _set_nested(config, ["aio_config", "queue_depth"], args.aio_queue_depth)
    _set_nested(config, ["aio_config", "thread_count"], args.aio_thread_count)
    _set_nested(config, ["aio_config", "block_size"], args.aio_block_size)
    _set_nested(config, ["aio_config", "single_submit"], args.aio_single_submit)
    _set_nested(config, ["aio_config", "pipeline_read"], args.aio_pipeline_read)
    _set_nested(config, ["aio_config", "pipeline_write"], args.aio_pipeline_write)

    runtime_config_output = args.runtime_config_output
    if runtime_config_output is None:
        os.makedirs("logs/runtime_configs", exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="ratel_",
            suffix=".json",
            dir="logs/runtime_configs",
            delete=False,
        )
        runtime_config_output = handle.name
    else:
        os.makedirs(os.path.dirname(os.path.abspath(runtime_config_output)), exist_ok=True)
        handle = open(runtime_config_output, "w")

    with handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    args.sb_config = os.path.abspath(runtime_config_output)
    print(f"RatelRuntimeConfig path={args.sb_config}", flush=True)
    print("RatelRuntimeConfigJSON " + json.dumps(config, sort_keys=True), flush=True)

def test_async(mp_queue_fp32, mp_queue_fp32_grad, mp_queue_signal, mp_queue_fp32_state_step, mp_queue_fp32_state_m, mp_queue_fp32_state_v, mp_model_parameters, mp_queue_fp32_state_id, mp_grad_event, mp_finish):
    model = torch.nn.Linear(10, 10)
    model_parameters = model.parameters()
    
    optimizer_parameters = {}
    optimizer = DeepSpeedCPUAdam(model_parameters, **optimizer_parameters, adamw_mode=False)
    
    @nvtx_wrap
    def cpu_step():
        optimizer.step()
    count = 0
    while(1):
        if not mp_finish.empty():
            if mp_finish.get() == 'finish':
                break
        if not mp_queue_fp32_state_id.empty():
            sub_group_id = mp_queue_fp32_state_id.get()

            # print(f'sub process get single {temp_signal}')
            temp_event = mp_grad_event.get()
            # print('bef sync', temp_event.query())
            temp_event.synchronize()
            # print('aft sync',temp_event.query())
            fp32_param = mp_queue_fp32.get()
            fp32_param.grad = mp_queue_fp32_grad.get()
            optimizer.state[fp32_param]['step'] = mp_queue_fp32_state_step.get()
            optimizer.state[fp32_param]['exp_avg'] = mp_queue_fp32_state_m.get()
            optimizer.state[fp32_param]['exp_avg_sq'] = mp_queue_fp32_state_v.get()

            optimizer.param_groups[0]['params'] = [fp32_param]
            # print(sub_group_id, optimizer.state[fp32_param])

            worker_start = time()
            cpu_step()
            worker_time = time() - worker_start
            print(f"RatelWorkerUpdate sub_group={sub_group_id} update_worker_s={worker_time:.6f}", flush=True)

            optimizer.param_groups[0]['params'] = []
            mp_queue_signal.put(sub_group_id)

            count += 1
            # print('finish')

if __name__ == '__main__':
    # 多进程初始化
    mp.set_start_method('spawn', force=True)
    mp_queue_fp32 = mp.Queue()
    mp_queue_fp32_grad = mp.Queue()
    mp_queue_signal = mp.Queue()
    mp_queue_fp32_state_step = mp.Queue()
    mp_queue_fp32_state_m = mp.Queue()
    mp_queue_fp32_state_v = mp.Queue()
    mp_queue_fp32_state_id = mp.Queue()
    mp_model_parameters = mp.Queue()
    mp_grad_event = mp.Queue()
    mp_finish = mp.Queue()
    mp_list = []
    mp_list.append(mp_queue_fp32)
    mp_list.append(mp_queue_fp32_grad)
    mp_list.append(mp_queue_signal)
    mp_list.append(mp_queue_fp32_state_step)
    mp_list.append(mp_queue_fp32_state_m)
    mp_list.append(mp_queue_fp32_state_v)
    mp_list.append(mp_queue_fp32_state_id) # 6
    mp_list.append(mp_grad_event)

    ## 解析参数
    # 解析模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default=None, choices=[None, '1.3B', '3B', '7B', '13B', '30B', '66B'], help="model size")
    parser.add_argument("--model_name", type=str, default=None, help="HF/local model path for Qwen/Qwen3/OPT runs.")
    parser.add_argument("--hidden_dim", type=int, default=5120, help="hidden dimension of transformer model")
    parser.add_argument("--num_heads", type=int, default=80, help="number of attention heads in transformer model")
    parser.add_argument("--num_layers", type=int, default=40, help="number of layers in transformer model")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="max sequence length")
    parser.add_argument("--vocab_size", type=int, default=50257, help="vocabulary size")
    parser.add_argument("--head_chunk_size", type=int, default=16, help="Qwen lm_head micro chunk size")
    parser.add_argument("--steps", type=int, default=10, help="number of measured iterations")
    parser.add_argument("--warmup_steps", type=int, default=0, help="iterations to run before averaging")
    parser.add_argument("--last_n", type=int, default=5, help="tail window for final averages")
    parser.add_argument("--stats_detail", action="store_true", help="print detailed JSON stats in addition to compact phase summaries")
    
    # 解析swap和重计算配置
    parser.add_argument("--is_swap_and_recompute", type=int, default=0, help="whether to use swap and recompute")
    parser.add_argument("--is_swap_prior", type=int, default=1, help="whether to consider swap prioritization")
    parser.add_argument("--is_fully_swap", type=int, default=0, help="whether to fully swap")
    parser.add_argument("--swap_ratio", type=float, default=0.2, help="swap ratio")

    # 解析异步和nvme配置
    parser.add_argument("--is_new_param_async", type=int, default=1, help="whether parameters are transmitted asynchronously")
    parser.add_argument("--is_grad_async", type=int, default=1, help="whether gradient are transmitted asynchronously")
    parser.add_argument("--is_mp", type=int, default=1, help="whether to use multiprocessing")
    parser.add_argument("--is_nvme", type=int, default=1, help="whether to offload to nvme")
    parser.add_argument("--is_nvme_async", type=int, default=1, help="whether to offload to nvme asynchronously")
    parser.add_argument("--is_nvme_rearrange", type=int, default=1, help="whether to reprogram nvme communications")

    parser.add_argument("--sb_config", type=str, default='/home/lcy/flush/Ratel_Private/config.json', help="config path")
    parser.add_argument("--runtime_config_output", type=str, default=None, help="where to write the resolved Ratel config")
    parser.add_argument("--activation_offload_device", choices=["cpu", "nvme"], default="nvme")
    parser.add_argument("--nvme_path", type=str, default=None)
    parser.add_argument("--param_buffer_count", type=int, default=None)
    parser.add_argument("--param_buffer_size", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--param_max_in_cpu", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--optimizer_buffer_count", type=int, default=None)
    parser.add_argument("--activation_buffer_count", type=int, default=None)
    parser.add_argument("--activation_buffer_size", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--activation_max_in_cpu", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--aio_queue_depth", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--aio_thread_count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--aio_block_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--aio_single_submit", type=lambda value: str(value).lower() in {"1", "true", "yes", "on"}, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--aio_pipeline_read", type=lambda value: str(value).lower() in {"1", "true", "yes", "on"}, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--aio_pipeline_write", type=lambda value: str(value).lower() in {"1", "true", "yes", "on"}, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.last_n <= 0:
        raise ValueError("last_n must be positive")
    if args.steps <= args.last_n + 3:
        raise ValueError(f"steps must be greater than last_n + 3 for benchmark statistics: steps={args.steps} last_n={args.last_n}")
    _prepare_runtime_config(args)

    model_kind, config, model_factory, act_stream, set_training = _configure_model(args)
    print_object(args, 'args')

    set_training(args)

    # 初始化矩阵乘激活值的优先级
    act_list = [i for i in range(4 * args.num_layers)]
    if args.is_swap_prior:
        act_priority = priority_sort(act_list)
    else:
        act_priority = act_list
    act_pack = {}
    print(act_priority)

    # 初始化模型，SSD-CPU-GPU三级存储初始化，参数属性改造
    see_memory_usage("before act ini")
    fw_time = []
    swap_list = []
    see_memory_usage("before model init")
    with Init(is_nvme=args.is_nvme, is_nvme_async=args.is_nvme_async, config=args.sb_config):
        model = model_factory(config)

    # 多进程初始化
    if args.is_mp:
        model.share_memory()
        # mp_model_parameters.put(model)
        p1 = mp.Process(target = test_async, args=(mp_queue_fp32, mp_queue_fp32_grad, mp_queue_signal, mp_queue_fp32_state_step, mp_queue_fp32_state_m, mp_queue_fp32_state_v, mp_model_parameters, mp_queue_fp32_state_id, mp_grad_event, mp_finish))
        p1.start()

    # Hook逻辑，实现参数异步预取和释放
    SB_hook(model, args.is_new_param_async, fw_time=fw_time, is_swap_and_recompute=args.is_swap_and_recompute)

    # 初始化输入和target, loss_fn
    hf_model = model_kind.startswith("qwen3") or model_kind == "hf_opt"
    input_shape = (args.batch_size, args.max_seq_len) if hf_model else (args.max_seq_len, args.batch_size)
    input_data = torch.randint(0, args.vocab_size, input_shape)
    input_data = input_data.to('cuda')
    target_shape = (
        (args.batch_size, args.max_seq_len, args.vocab_size)
        if hf_model
        else (args.max_seq_len, args.batch_size, args.hidden_dim)
    )
    target = torch.randn(*target_shape, dtype=torch.float16)
    target = target.to('cuda')
    loss_fn = nn.MSELoss()

    # 初始化CPU Adam，和优化器相关
    model_parameters = model.parameters()
    optimizer_parameters = {}
    optimizer = DeepSpeedCPUAdam(model_parameters,
                                        **optimizer_parameters,
                                        adamw_mode=False)    
    
    # 改造优化器，实现异步梯度卸载和异步优化器更新
    optimizer = SB_optimizer(optimizer, args.is_mp, mp_list = mp_list, is_nvme=args.is_nvme, is_grad_async=args.is_grad_async, is_nvme_async=args.is_nvme_async, is_nvme_rearrange=args.is_nvme_rearrange, config=args.sb_config)

    fwd_time_list=[]
    bck_time_list=[]
    event_list = []
    reset_cuda_peaks()
    for i in range(args.steps + args.warmup_steps):
        iter_start = time()
        print(f'-----------------------Iter {i}-----------------------')
        print('---begin forward---')
        torch.cuda.nvtx.range_push("iteration")
        
        torch.cuda.nvtx.range_push("forward")
        with phase("forward"):
            if hf_model:
                output = model(input_data).last_hidden_state
            else:
                output = model(input_data, swap_list, act_pack)
        torch.cuda.nvtx.range_pop()
        
        # 自动调度swap和重计算
        if i == 0 and args.is_swap_and_recompute:
            get_act_swap_list(fw_time, args, swap_list, act_pack, act_priority)

        torch.cuda.current_stream().synchronize()
        act_stream.synchronize()
        forward_end = time()
        print('forward time', forward_end - iter_start)
        forward_time = forward_end - iter_start
        if i >= args.warmup_steps:
            fwd_time_list.append(forward_time)

        if hf_model:
            losses = []
            chunk_size = args.head_chunk_size
            for chunk_start in range(0, args.batch_size, chunk_size):
                chunk_output = output[chunk_start:chunk_start + chunk_size]
                chunk_target = target[chunk_start:chunk_start + chunk_size]
                losses.append(loss_fn(model.lm_head(chunk_output), chunk_target))
            loss = torch.stack(losses).mean()
        else:
            loss = loss_fn(output, target)

        print('---begin backward---')
        torch.cuda.nvtx.range_push("backward")
        with phase("backward"):
            loss.backward()
            optimizer.independent_gradient_partition_epilogue()
        backward_end = time()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("optimizer")

        update_start = time()
        with phase("update"):
            if not args.is_mp and not args.is_nvme_async:
                optimizer.step()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        global_back_id = 0
        event_list = []
        global_flag_id = 0
        torch.cuda.current_stream().synchronize()
        update_end = time()
        backward_time = backward_end - forward_end
        update_time = update_end - update_start
        print('backward time', backward_time)
        print('update time', update_time)
        print('back_and_opt time', update_end - forward_end)
        if i >= args.warmup_steps:
            bck_time_list.append(backward_time)
            record_step(
                step=i - args.warmup_steps,
                forward_s=forward_time,
                backward_s=backward_time,
                update_s=update_time,
                total_s=update_end - iter_start,
            )

    import numpy as np
    tail = args.last_n
    avg_fwd_time = np.mean(fwd_time_list[:-1][-tail:])
    avg_bck_time = np.mean(bck_time_list[:-1][-tail:])
    print(f"平均前向时间是{avg_fwd_time}")
    print(f"平均反向时间是{avg_bck_time}")
    print(f"平均epoch时间是{avg_fwd_time + avg_bck_time}")
    print_summary(last_n=args.last_n, detail=args.stats_detail)
    
    torch.cuda.current_stream().synchronize()
    mp_finish.put('finish')
    if args.is_mp:
        p1.join()



