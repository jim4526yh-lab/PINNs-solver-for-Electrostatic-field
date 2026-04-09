import torch
# 检查GPU是否可用
if torch.cuda.is_available():
   device_index = torch.cuda.current_device()
   device_count = torch.cuda.device_count()
   print("GPU可用！\n当前设备索引:{0}\nGPU数量: {1}\n".format(device_index, device_count))
   for cnt in range(device_count):
      device_name = torch.cuda.get_device_name(cnt)
      print("GPU{0} 信息: {1}\n".format(cnt, device_name))
else:
   print("GPU不可用，将使用CPU进行计算。\n")
