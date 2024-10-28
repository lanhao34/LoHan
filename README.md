# Ratel

This artifact provides a guid to replicate the primary experiments in this paper. You can follow this repository to reproduce the experimental results about Ratel's maximum trainable model sizes, batch sizes and throughput in our paper. The documentation and auto-run script mainly focus on reproducing results in Subsection V-B and you can adjust the code to reproduce results in other sections. 

## Environment Setup

### SSD Configuration

Ratel aggregates the I/O bandwidth of multiple SSDs by configuring a RAID array for efficient model states and activation offloading. Therefore, we provide a script to configure this array.

First, modify the `make_raid.sh` to meet your own need. The script in this repo is used to configure the drives `/dev/nvme0n1` to `/dev/nvme11n1` into an array. You can adjust the line 23 to change the drives you want to set up.

After configuring the script, you can run the script to set up the RAID array. You might need a root permission to do so:

```shell
./make_raid.sh
```

### Installing the Python packages

```shell
conda create -n ratel python=3.10
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# If there has different CUDA version, you should specify the CUDA version
# export CUDA_HOME=/usr/local/cuda-11.8
pip install flash-attn==1.0.4

# The following two packages are to fulfill the requirements of the ogb
pip install six==1.16.0
pip install scikit-learn
```

## Running Ratel

We provide a script to run Ratel. You can adjust the script to reproduce the results. 

```shell
bash run.sh
```

### Limiting the Memory Size

Experiments in Subsection V-B requires adjusting the main memory capacity. Instead of manually adding and removing the machine's DRAM, you can consider pinning the main memory via huge page so that these memory spaces cannot be utilized by Ratel. 

You can use the following script (root permission required) to pin the main memory

```shell
sh -c "echo 1024 > /proc/sys/vm/nr_hugepages"
```

The **1024** means you set the **HugePages_num** to **1024**. Each HugePageSize is **2MB**. Therefore the total memory size pinned by huge page in this script is **2MB * 1024 = 2048MB**.

You can check the pinned memory by using the following command.

```shell
$ cat /proc/meminfo | grep Huge
```

For example, the following output indicates the memory pinned bu huge page is 1024(HugePages_Total)*2048kB(Hugepagesize)=2GB.

```
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:    1024
HugePages_Free:     1024
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:         2097152 kB
```

### Benchmark Results

The results are produced under the testbed whose configuration is listed below.

| CPU         | Dual Intel Xeon Gold 5320 CPU               |
|-------------|---------------------------------------------|
| Main Memory | 768 GB 3200 MHz DDR4 (16 channels in total) |
| GPU         | NVIDIA GeForce RTX 4090                     |
| SSD         | 12x D7-P5510 3.84 TB SSD                    |

In the following, we list our major experimental results for reference (while other figures generally reuse these experimental data when evaluating Ratel). All the throughput data are measured in TFLOPS (which is directly outputed by the script. )

---

Figure 5(a)/7(a): End-to-end performance, single 4090 GPU.

Model Configuration:

| #Params            | #Layers | #Heads | #Hidden Dimemsion |
|--------------------|---------|--------|-------------------|
| 1.3$\times10^{10}$ | 40      | 40     | 5120              |

Result: 

| Batch Size | 8    | 16   | 32    | 64    | 128   |
|------------|------|------|-------|-------|-------|
| TFLOPS     | 42.8 | 84.3 | 143.1 | 155.8 | 153.8 |

---

Figure 7(b): End-to-end performance, single 4090 GPU. 

Model Configuration:

| #Params             | #Layers | #Heads | #Hidden Dimemsion |
|---------------------|---------|--------|-------------------|
| 1.75$\times10^{11}$ | 96      | 96     | 12288             |

Result: 

| Batch Size | 8    | 16   | 32  |
|------------|------|------|-----|
| TFLOPS     | 52.6 | 86.9 | OOM |

---

Figure 10(b): Throughput w.r.t. number of SSDs, single 4090 GPU. 

Model Configuration:

| #Params            | #Layers | #Heads | #Hidden Dimemsion |
|--------------------|---------|--------|-------------------|
| 1.3$\times10^{10}$ | 40      | 40     | 5120              |

Result: 

| #SSDs  | 1    | 2     | 3     | 6     | 12    |
|--------|------|-------|-------|-------|-------|
| bsz=32 | 37.5 | 64.3  | 81.1  | 121.7 | 142.0 |
| bsz=48 | 53.1 | 89.7  | 121.7 | 146.3 | 153.9 |
| bsz=64 | 70.3 | 111.7 | 136.3 | 151.5 | 148.2 |

---

Figure 11(a): End-to-end performance, 2x 4090 GPUs. 

Model Configuration:

| #Params            | #Layers | #Heads | #Hidden Dimemsion |
|--------------------|---------|--------|-------------------|
| 1.3$\times10^{10}$ | 40      | 40     | 5120              |

Result: 

| Global Batch Size | 16   | 32    | 64    | 128   |
|-------------------|------|-------|-------|-------|
| Global TFLOPS     | 55.0 | 103.2 | 194.7 | 278.2 |

---

Figure 11(b): End-to-end performance, 2x 4090 GPUs. 

Model Configuration:

| #Params          | #Layers | #Heads | #Hidden Dimemsion |
|------------------|---------|--------|-------------------|
| 7$\times10^{10}$ | 80      | 64     | 8192              |

Result: 

| Global Batch Size | 16   | 32    | 48    |
|-------------------|------|-------|-------|
| Global TFLOPS     | 64.6 | 128.8 | 183.5 |

---

Figure 11(c): End-to-end performance, 4x 4090 GPUs. 

Model Configuration:

| #Params            | #Layers | #Heads | #Hidden Dimemsion |
|--------------------|---------|--------|-------------------|
| 1.3$\times10^{10}$ | 40      | 40     | 5120              |

Result: 

| Global Batch Size | 32    | 64    | 128   | 256   |
|-------------------|-------|-------|-------|-------|
| Global TFLOPS     | 106.5 | 209.7 | 358.7 | 514.4 |

---

Figure 11(d): End-to-end performance, 4x 4090 GPUs. 

Model Configuration:

| #Params          | #Layers | #Heads | #Hidden Dimemsion |
|------------------|---------|--------|-------------------|
| 7$\times10^{10}$ | 80      | 64     | 8192              |

Result: 

| Global Batch Size | 32    | 64    | 96    |
|-------------------|-------|-------|-------|
| Global TFLOPS     | 124.8 | 249.7 | 348.4 |

## Acknowledgement

Some of the code in this project is modified from the [DeepSpeed](https://github.com/microsoft/DeepSpeed) repository, we appreciate the contributions of the original repository authors.

* op_ds/accelerator/real_accelerator.py
* op_ds/ops/op_builder/all_ops.py
* op_ds/ops/op_builder/builder.py
* op_ds/ops/CPUAdam.py
* nvme_ds
