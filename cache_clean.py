import torch
import psutil
import gc
from numba import cuda

if __name__ == "__main__":
	# torch.cuda.empty_cache()
	# 获取系统内存占用情况
	memory_info = psutil.virtual_memory()
	
	# 获取总内存大小（以字节为单位）
	total_memory = memory_info.total
	
	# 获取已使用的内存大小（以字节为单位）
	used_memory = memory_info.used
	
	# 获取空闲内存大小（以字节为单位）
	free_memory = memory_info.available
	
	# 打印内存使用情况
	print(f"Total Memory: {total_memory / (1024 * 1024 * 1024)} GB")
	print(f"Used Memory: {used_memory / (1024 * 1024 * 1024)} GB")
	print(f"Free Memory: {free_memory / (1024 * 1024 * 1024)} GB")

	gc.collect()
	cuda.select_device(0)
	cuda.close()
	cuda.select_device(0)
	torch.cuda.memory_summary()
	for obj in gc.get_objects():
		if torch.is_tensor(obj) and obj.is_cuda:
			print(f"残留张量: {type(obj)}, 大小: {obj.size()}")