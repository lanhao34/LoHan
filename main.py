from ratel_init import Init, SB_hook
from ratel_optimizer import SB_optimizer
from see_mem import see_memory_usage
from nvtx import nvtx_wrap
import argparse
import torch
import torch.nn as nn
from op_ds.ops.CPUAdam import DeepSpeedCPUAdam
import torch.multiprocessing as mp
from time import time
from gpt_model import GPT2Model, GPT2Config, act_stream, set_training
from utils import priority_sort, get_act_swap_list
from nvme_ds.utils import print_object

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

            cpu_step()

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
    parser.add_argument("--hidden_dim", type=int, default=5120, help="hidden dimension of transformer model")
    parser.add_argument("--num_heads", type=int, default=80, help="number of attention heads in transformer model")
    parser.add_argument("--num_layers", type=int, default=40, help="number of layers in transformer model")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="max sequence length")
    parser.add_argument("--vocab_size", type=int, default=50257, help="vocabulary size")
    
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
    args = parser.parse_args()

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
    print_object(args, 'args')


    # 初始化模型config
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

    set_training(args)

    # 初始化矩阵乘激活值的优先级
    act_list = [i for i in range(4 * config.num_layers)]
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
        model = GPT2Model(config).half()

    # 多进程初始化
    if args.is_mp:
        model.share_memory()
        # mp_model_parameters.put(model)
        p1 = mp.Process(target = test_async, args=(mp_queue_fp32, mp_queue_fp32_grad, mp_queue_signal, mp_queue_fp32_state_step, mp_queue_fp32_state_m, mp_queue_fp32_state_v, mp_model_parameters, mp_queue_fp32_state_id, mp_grad_event, mp_finish))
        p1.start()

    # Hook逻辑，实现参数异步预取和释放
    SB_hook(model, args.is_new_param_async, fw_time=fw_time, is_swap_and_recompute=args.is_swap_and_recompute)

    # 初始化输入和target, loss_fn
    input_data = torch.randint(0, args.vocab_size, (args.max_seq_len, args.batch_size))
    input_data = input_data.to('cuda')
    target = torch.randn(args.max_seq_len, args.batch_size, args.hidden_dim, dtype=torch.float16)
    target = target.to('cuda')
    loss_fn = nn.MSELoss()

    # 初始化CPU Adam，和优化器相关
    model_parameters = model.parameters()
    embed_params = list(model.token_emb.parameters()) + list(model.pos_emb.parameters()) + list(model.drop.parameters())
    sub_group_size = sum([p.ds_numel for p in embed_params])
    optimizer_parameters = {}
    optimizer = DeepSpeedCPUAdam(model_parameters,
                                        **optimizer_parameters,
                                        adamw_mode=False)    
    
    # 改造优化器，实现异步梯度卸载和异步优化器更新
    optimizer = SB_optimizer(optimizer, args.is_mp, mp_list = mp_list, is_nvme=args.is_nvme, is_grad_async=args.is_grad_async, is_nvme_async=args.is_nvme_async, is_nvme_rearrange=args.is_nvme_rearrange, config=args.sb_config)

    fwd_time_list=[]
    bck_time_list=[]
    event_list = []
    for i in range(10):
        iter_start = time()
        print(f'-----------------------Iter {i}-----------------------')
        print('---begin forward---')
        torch.cuda.nvtx.range_push("iteration")
        
        torch.cuda.nvtx.range_push("forward")
        output = model(input_data, swap_list, act_pack)
        torch.cuda.nvtx.range_pop()
        
        # 自动调度swap和重计算
        if i == 0 and args.is_swap_and_recompute:
            get_act_swap_list(fw_time, args, swap_list, act_pack, act_priority)

        torch.cuda.current_stream().synchronize()
        act_stream.synchronize()
        forward_end = time()
        print('forward time', forward_end - iter_start)
        fwd_time_list.append(forward_end - iter_start)

        loss = loss_fn(output, target)

        print('---begin backward---')
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        optimizer.independent_gradient_partition_epilogue()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("optimizer")

        if not args.is_mp and not args.is_nvme_async:
            optimizer.step()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        global_back_id = 0
        event_list = []
        global_flag_id = 0
        torch.cuda.current_stream().synchronize()
        print('back_and_opt time', time() - forward_end)
        bck_time_list.append(time() - forward_end)

    import numpy as np
    avg_fwd_time = np.mean(fwd_time_list[-5:])
    avg_bck_time = np.mean(bck_time_list[-5:])
    print(f"平均前向时间是{avg_fwd_time}")
    print(f"平均反向时间是{avg_bck_time}")
    print(f"平均epoch时间是{avg_fwd_time + avg_bck_time}")
    
    torch.cuda.current_stream().synchronize()
    mp_finish.put('finish')
    if args.is_mp:
        p1.join()



