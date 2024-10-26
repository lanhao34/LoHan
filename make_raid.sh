#!/bin/bash

# 列出/dev目录下所有以md开头的设备
devices=$(ls -1 /dev/md* 2>/dev/null)

# 检查是否有符合条件的设备
if [ -z "$devices" ]; then
  echo "没有找到符合条件的设备。"
  exit 1
fi

# 卸载设备和停用 RAID
for device in $devices; do
  echo "卸载设备：$device"
  umount "$device" 2>/dev/null
  mdadm --stop "$device" 2>/dev/null
done

echo "所有设备已卸载并停用。"

umount /dev/nvme0n1

mdadm --misc --zero-superblock /dev/nvme{0,1,2,3,4,5,6,7,8,9,10,11}n1

index_num=12
start_num=0
end_num=11

# 计算设备数量
num_devices=$((end_num - start_num + 1))

# 检查设备数量是否为正数
if [ "$num_devices" -lt 1 ]; then
  echo "无效的设备范围。"
  exit 1
fi

# 创建RAID组
echo "创建包含 $num_devices 个nvme设备的RAID组"
mdadm --create "/dev/md$index_num" --auto=yes --level=0 -n "$num_devices" $(eval echo /dev/nvme{${start_num}..${end_num}}n1)

# 格式化RAID设备
mkfs.ext4 "/dev/md$index_num"

# 创建挂载目录
mount_dir="/share/data$num_devices"
mkdir -p "$mount_dir"

# 挂载RAID设备
mount "/dev/md$index_num" "$mount_dir"

echo "RAID已创建并挂载到 $mount_dir"

chmod 777 "$mount_dir"

echo "$mount_dir权限已更改"
