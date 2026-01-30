"""
技术指标基类
提供 GPU 加速的指标计算框架
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List
import numpy as np

# GPU 加速库
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    cuda = None
    jit = None
    NUMBA_AVAILABLE = False


class GPUAccelerator:
    """
    GPU 加速器
    
    管理 GPU 资源和自动选择计算后端。
    """
    
    def __init__(self, device_id: int = 0, force_cpu: bool = False):
        """
        初始化加速器
        
        Args:
            device_id: GPU 设备 ID
            force_cpu: 强制使用 CPU
        """
        self.device_id = device_id
        self.force_cpu = force_cpu
        self._use_gpu = GPU_AVAILABLE and not force_cpu
        
        if self._use_gpu:
            try:
                cp.cuda.Device(device_id).use()
            except Exception as e:
                print(f"GPU 初始化失败: {e}, 回退到 CPU")
                self._use_gpu = False
    
    @property
    def use_gpu(self) -> bool:
        """是否使用 GPU"""
        return self._use_gpu
    
    @property
    def xp(self):
        """获取当前计算库（cupy 或 numpy）"""
        return cp if self._use_gpu else np
    
    def to_device(self, data: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
        """将数据传输到计算设备"""
        if self._use_gpu:
            return cp.asarray(data)
        return data
    
    def to_host(self, data) -> np.ndarray:
        """将数据传回 CPU"""
        if self._use_gpu and hasattr(data, 'get'):
            return data.get()
        return np.asarray(data)
    
    def synchronize(self) -> None:
        """同步 GPU 计算"""
        if self._use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    def memory_info(self) -> Dict[str, Any]:
        """获取 GPU 内存信息"""
        if not self._use_gpu:
            return {"gpu_available": False}
        
        mempool = cp.get_default_memory_pool()
        return {
            "gpu_available": True,
            "device_id": self.device_id,
            "used_bytes": mempool.used_bytes(),
            "total_bytes": mempool.total_bytes(),
        }
    
    def clear_cache(self) -> None:
        """清理 GPU 内存缓存"""
        if self._use_gpu:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()


# 全局加速器实例
_default_accelerator: Optional[GPUAccelerator] = None


def get_accelerator(device_id: int = 0) -> GPUAccelerator:
    """获取或创建全局加速器"""
    global _default_accelerator
    if _default_accelerator is None:
        _default_accelerator = GPUAccelerator(device_id)
    return _default_accelerator


class Indicator(ABC):
    """
    技术指标基类
    
    提供 GPU 加速的指标计算框架。
    子类需实现 compute 方法。
    """
    
    def __init__(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        accelerator: Optional[GPUAccelerator] = None,
    ):
        """
        初始化指标
        
        Args:
            name: 指标名称
            params: 指标参数
            accelerator: GPU 加速器
        """
        self.name = name
        self.params = params or {}
        self.accelerator = accelerator or get_accelerator()
        self._cache: Dict[str, Any] = {}
    
    @property
    def xp(self):
        """获取计算库"""
        return self.accelerator.xp
    
    @abstractmethod
    def compute(self, data: np.ndarray) -> np.ndarray:
        """
        计算指标
        
        Args:
            data: 输入数据（通常是收盘价数组）
            
        Returns:
            指标值数组
        """
        pass
    
    def compute_batch(self, data_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        批量计算指标（利用 GPU 并行）
        
        Args:
            data_list: 数据列表
            
        Returns:
            指标值列表
        """
        if self.accelerator.use_gpu:
            # GPU 批量计算
            results = []
            for data in data_list:
                result = self.compute(data)
                results.append(result)
            return results
        else:
            # CPU 顺序计算
            return [self.compute(data) for data in data_list]
    
    def _to_device(self, data: np.ndarray):
        """将数据传输到计算设备"""
        return self.accelerator.to_device(data)
    
    def _to_host(self, data) -> np.ndarray:
        """将数据传回 CPU"""
        return self.accelerator.to_host(data)
    
    def clear_cache(self) -> None:
        """清理缓存"""
        self._cache.clear()
    
    def __repr__(self):
        return f"Indicator({self.name}, params={self.params})"


class SMA(Indicator):
    """
    简单移动平均（GPU 加速）
    """
    
    def __init__(
        self,
        period: int = 20,
        accelerator: Optional[GPUAccelerator] = None,
    ):
        super().__init__("SMA", {"period": period}, accelerator)
        self.period = period
    
    def compute(self, data: np.ndarray) -> np.ndarray:
        """计算 SMA"""
        xp = self.xp
        data = self._to_device(data)
        
        if len(data) < self.period:
            return self._to_host(xp.array([]))
        
        # 使用卷积计算移动平均
        kernel = xp.ones(self.period) / self.period
        sma = xp.convolve(data, kernel, mode='valid')
        
        return self._to_host(sma)


class EMA(Indicator):
    """
    指数移动平均（GPU 加速）
    """
    
    def __init__(
        self,
        period: int = 20,
        accelerator: Optional[GPUAccelerator] = None,
    ):
        super().__init__("EMA", {"period": period}, accelerator)
        self.period = period
        self.alpha = 2 / (period + 1)
    
    def compute(self, data: np.ndarray) -> np.ndarray:
        """计算 EMA"""
        xp = self.xp
        data = self._to_device(data)
        
        if len(data) < self.period:
            return self._to_host(xp.array([]))
        
        # EMA 计算（需要顺序计算，GPU 优势有限）
        ema = xp.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = self.alpha * data[i] + (1 - self.alpha) * ema[i - 1]
        
        return self._to_host(ema)


class RSI(Indicator):
    """
    相对强弱指标（GPU 加速）
    """
    
    def __init__(
        self,
        period: int = 14,
        accelerator: Optional[GPUAccelerator] = None,
    ):
        super().__init__("RSI", {"period": period}, accelerator)
        self.period = period
    
    def compute(self, data: np.ndarray) -> np.ndarray:
        """计算 RSI"""
        xp = self.xp
        data = self._to_device(data)
        
        if len(data) < self.period + 1:
            return self._to_host(xp.array([]))
        
        # 计算价格变动
        deltas = xp.diff(data)
        gains = xp.where(deltas > 0, deltas, 0)
        losses = xp.where(deltas < 0, -deltas, 0)
        
        # 计算平均涨跌幅
        avg_gain = xp.zeros(len(deltas))
        avg_loss = xp.zeros(len(deltas))
        
        avg_gain[self.period - 1] = xp.mean(gains[:self.period])
        avg_loss[self.period - 1] = xp.mean(losses[:self.period])
        
        for i in range(self.period, len(deltas)):
            avg_gain[i] = (avg_gain[i - 1] * (self.period - 1) + gains[i]) / self.period
            avg_loss[i] = (avg_loss[i - 1] * (self.period - 1) + losses[i]) / self.period
        
        rs = xp.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        return self._to_host(rsi[self.period - 1:])


class MACD(Indicator):
    """
    MACD 指标（GPU 加速）
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        accelerator: Optional[GPUAccelerator] = None,
    ):
        super().__init__(
            "MACD",
            {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
            },
            accelerator,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def compute(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算 MACD
        
        Returns:
            包含 macd, signal, histogram 的字典
        """
        xp = self.xp
        data = self._to_device(data)
        
        if len(data) < self.slow_period:
            empty = self._to_host(xp.array([]))
            return {"macd": empty, "signal": empty, "histogram": empty}
        
        # 计算快慢 EMA
        fast_alpha = 2 / (self.fast_period + 1)
        slow_alpha = 2 / (self.slow_period + 1)
        
        fast_ema = xp.zeros(len(data))
        slow_ema = xp.zeros(len(data))
        fast_ema[0] = data[0]
        slow_ema[0] = data[0]
        
        for i in range(1, len(data)):
            fast_ema[i] = fast_alpha * data[i] + (1 - fast_alpha) * fast_ema[i - 1]
            slow_ema[i] = slow_alpha * data[i] + (1 - slow_alpha) * slow_ema[i - 1]
        
        macd_line = fast_ema - slow_ema
        
        # 计算信号线
        signal_alpha = 2 / (self.signal_period + 1)
        signal_line = xp.zeros(len(data))
        signal_line[0] = macd_line[0]
        
        for i in range(1, len(data)):
            signal_line[i] = signal_alpha * macd_line[i] + (1 - signal_alpha) * signal_line[i - 1]
        
        histogram = macd_line - signal_line
        
        return {
            "macd": self._to_host(macd_line),
            "signal": self._to_host(signal_line),
            "histogram": self._to_host(histogram),
        }


class BollingerBands(Indicator):
    """
    布林带指标（GPU 加速）
    """
    
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        accelerator: Optional[GPUAccelerator] = None,
    ):
        super().__init__(
            "BollingerBands",
            {"period": period, "std_dev": std_dev},
            accelerator,
        )
        self.period = period
        self.std_dev = std_dev
    
    def compute(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算布林带
        
        Returns:
            包含 upper, middle, lower 的字典
        """
        xp = self.xp
        data = self._to_device(data)
        
        if len(data) < self.period:
            empty = self._to_host(xp.array([]))
            return {"upper": empty, "middle": empty, "lower": empty}
        
        # 计算中轨（SMA）
        kernel = xp.ones(self.period) / self.period
        middle = xp.convolve(data, kernel, mode='valid')
        
        # 计算标准差
        std = xp.zeros(len(middle))
        for i in range(len(middle)):
            window = data[i:i + self.period]
            std[i] = xp.std(window)
        
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        
        return {
            "upper": self._to_host(upper),
            "middle": self._to_host(middle),
            "lower": self._to_host(lower),
        }


class ATR(Indicator):
    """
    平均真实波幅（GPU 加速）
    """
    
    def __init__(
        self,
        period: int = 14,
        accelerator: Optional[GPUAccelerator] = None,
    ):
        super().__init__("ATR", {"period": period}, accelerator)
        self.period = period
    
    def compute_with_hlc(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """
        计算 ATR
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            
        Returns:
            ATR 数组
        """
        xp = self.xp
        high = self._to_device(high)
        low = self._to_device(low)
        close = self._to_device(close)
        
        if len(high) < self.period + 1:
            return self._to_host(xp.array([]))
        
        # 计算真实波幅
        prev_close = xp.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = xp.abs(high - prev_close)
        tr3 = xp.abs(low - prev_close)
        
        tr = xp.maximum(xp.maximum(tr1, tr2), tr3)
        
        # 计算 ATR（平滑移动平均）
        atr = xp.zeros(len(tr))
        atr[self.period - 1] = xp.mean(tr[:self.period])
        
        for i in range(self.period, len(tr)):
            atr[i] = (atr[i - 1] * (self.period - 1) + tr[i]) / self.period
        
        return self._to_host(atr[self.period - 1:])
    
    def compute(self, data: np.ndarray) -> np.ndarray:
        """简化版本，假设输入包含 HLC 数据"""
        raise NotImplementedError("ATR 需要 HLC 数据，请使用 compute_with_hlc 方法")
