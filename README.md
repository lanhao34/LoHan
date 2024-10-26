# Ratel_Private

## Quick start
```shell
bash run.sh
```

## Set the environment

```shell
conda create -n torch python=3.10
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# If there has different CUDA version, you should specify the CUDA version
# export CUDA_HOME=/usr/local/cuda-11.8
pip install flash-attn==1.0.4

# The following two packages are to fulfill the requirements of the ogb
pip install six==1.16.0
pip install scikit-learn
```

## Set the RAID

Some of the following command may need **sudo** permission.

```shell
sudo ./make_raid.sh
```

## Limit the memory size

By using the following command, the memory on your device will be occupied by the huge page.

```shell
sudo sh -c "echo 1024 > /proc/sys/vm/nr_hugepages"
```

The **1024** means you will set the **HugePages_num** to **1024**. And each HugePageSize is **2MB**. So the total memory size occupied will be **2MB * 1024 = 2048MB**.

You can get the related information by using the following command.

```shell
$ cat /proc/meminfo | grep Huge
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

And this means your available memory size will be **Total-2GB**.

## 

The following files from [DeepSpeed](https://github.com/microsoft/DeepSpeed) have been modified:

* myDeep/accelerator/real_accelerator.py
* myDeep/ops/op_builder/all_ops.py
* myDeep/ops/op_builder/builder.py
* myDeep/ops/CPUAdam.py
* nvme_ds