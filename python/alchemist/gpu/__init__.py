"""
GPU 工具模块
提供 GPU 相关的工具函数
"""

from typing import Optional, Dict, Any, List, Tuple
import os

# GPU 库
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def check_gpu_availability() -> Dict[str, Any]:
    """
    检查 GPU 可用性
    
    Returns:
        GPU 信息字典
    """
    result = {
        "cupy_available": CUPY_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "gpu_count": 0,
        "devices": [],
    }
    
    if CUPY_AVAILABLE:
        try:
            result["cuda_available"] = True
            result["gpu_count"] = cp.cuda.runtime.getDeviceCount()
            
            for i in range(result["gpu_count"]):
                props = cp.cuda.runtime.getDeviceProperties(i)
                result["devices"].append({
                    "id": i,
                    "name": props["name"].decode(),
                    "total_memory": props["totalGlobalMem"],
                    "total_memory_gb": props["totalGlobalMem"] / (1024**3),
                })
        except Exception as e:
            result["error"] = str(e)
    
    elif TORCH_AVAILABLE:
        try:
            result["cuda_available"] = torch.cuda.is_available()
            if result["cuda_available"]:
                result["gpu_count"] = torch.cuda.device_count()
                
                for i in range(result["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    result["devices"].append({
                        "id": i,
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "total_memory_gb": props.total_memory / (1024**3),
                    })
        except Exception as e:
            result["error"] = str(e)
    
    return result


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, Any]:
    """
    获取 GPU 内存信息
    
    Args:
        device_id: GPU 设备 ID
        
    Returns:
        内存信息字典
    """
    if CUPY_AVAILABLE:
        try:
            with cp.cuda.Device(device_id):
                mempool = cp.get_default_memory_pool()
                return {
                    "device_id": device_id,
                    "used_bytes": mempool.used_bytes(),
                    "total_bytes": mempool.total_bytes(),
                    "free_bytes": mempool.total_bytes() - mempool.used_bytes(),
                    "used_gb": mempool.used_bytes() / (1024**3),
                    "total_gb": mempool.total_bytes() / (1024**3),
                }
        except Exception as e:
            return {"error": str(e)}
    
    elif TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return {
                "device_id": device_id,
                "allocated_bytes": torch.cuda.memory_allocated(device_id),
                "reserved_bytes": torch.cuda.memory_reserved(device_id),
                "allocated_gb": torch.cuda.memory_allocated(device_id) / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved(device_id) / (1024**3),
            }
        except Exception as e:
            return {"error": str(e)}
    
    return {"error": "No GPU library available"}


def set_gpu_device(device_id: int) -> None:
    """
    设置当前 GPU 设备
    
    Args:
        device_id: GPU 设备 ID
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    if CUPY_AVAILABLE:
        cp.cuda.Device(device_id).use()
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.set_device(device_id)


def clear_gpu_memory(device_id: Optional[int] = None) -> None:
    """
    清理 GPU 内存
    
    Args:
        device_id: GPU 设备 ID（None 表示所有设备）
    """
    if CUPY_AVAILABLE:
        if device_id is not None:
            with cp.cuda.Device(device_id):
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
        else:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()


class MultiGPUManager:
    """
    多 GPU 管理器
    
    管理多个 GPU 设备的资源分配。
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        初始化管理器
        
        Args:
            device_ids: GPU 设备 ID 列表
        """
        gpu_info = check_gpu_availability()
        
        if device_ids is None:
            device_ids = list(range(gpu_info["gpu_count"]))
        
        self.device_ids = device_ids
        self.device_count = len(device_ids)
        self._current_device_index = 0
    
    def get_next_device(self) -> int:
        """
        获取下一个可用设备（轮询）
        
        Returns:
            设备 ID
        """
        device_id = self.device_ids[self._current_device_index]
        self._current_device_index = (self._current_device_index + 1) % self.device_count
        return device_id
    
    def get_least_used_device(self) -> int:
        """
        获取内存使用最少的设备
        
        Returns:
            设备 ID
        """
        min_usage = float("inf")
        best_device = self.device_ids[0]
        
        for device_id in self.device_ids:
            info = get_gpu_memory_info(device_id)
            usage = info.get("used_bytes", float("inf"))
            if usage < min_usage:
                min_usage = usage
                best_device = device_id
        
        return best_device
    
    def distribute_work(self, items: List[Any]) -> Dict[int, List[Any]]:
        """
        将工作分配到多个 GPU
        
        Args:
            items: 待处理项目列表
            
        Returns:
            {device_id: [items]} 字典
        """
        distribution = {device_id: [] for device_id in self.device_ids}
        
        for i, item in enumerate(items):
            device_id = self.device_ids[i % self.device_count]
            distribution[device_id].append(item)
        
        return distribution
    
    def summary(self) -> Dict[str, Any]:
        """获取管理器摘要"""
        devices_info = []
        for device_id in self.device_ids:
            info = get_gpu_memory_info(device_id)
            info["id"] = device_id
            devices_info.append(info)
        
        return {
            "device_count": self.device_count,
            "device_ids": self.device_ids,
            "devices": devices_info,
        }


def parallel_compute_on_gpus(
    func,
    data_list: List[Any],
    device_ids: Optional[List[int]] = None,
) -> List[Any]:
    """
    在多个 GPU 上并行计算
    
    Args:
        func: 计算函数
        data_list: 数据列表
        device_ids: GPU 设备 ID 列表
        
    Returns:
        结果列表
    """
    if not CUPY_AVAILABLE and not TORCH_AVAILABLE:
        # 无 GPU，顺序执行
        return [func(data) for data in data_list]
    
    manager = MultiGPUManager(device_ids)
    distribution = manager.distribute_work(list(enumerate(data_list)))
    
    results = [None] * len(data_list)
    
    for device_id, items in distribution.items():
        if CUPY_AVAILABLE:
            with cp.cuda.Device(device_id):
                for idx, data in items:
                    results[idx] = func(data)
        elif TORCH_AVAILABLE:
            with torch.cuda.device(device_id):
                for idx, data in items:
                    results[idx] = func(data)
    
    return results
