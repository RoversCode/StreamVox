"""
mel.py - 极致对齐的 Mel 谱图提取器 (纯 NumPy & SciPy 实现)
无需 librosa 依赖，完美对齐官方对 librosa.filters.mel 和 librosa.stft 的调用结果。
"""
import numpy as np
from scipy.signal.windows import hann
from scipy.fft import rfft

class MelExtractor:
    """
    Qwen3-TTS 专用的 Mel 提取逻辑。
    参数对齐: sr=24000, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000
    """
    def __init__(self, sr=24000, n_fft=1024, hop_length=256, n_mels=128, fmin=0.0, fmax=12000.0):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        # 1. 预计算 Slaney Mel Filterbank 权重
        self.mel_basis = self._build_mel_basis()
        
        # 2. 预计算窗函数 (periodic hann)
        self.window = hann(n_fft, sym=False).astype(np.float32)
    
    def _hz_to_mel(self, frequencies):
        frequencies = np.asanyarray(frequencies)
        f_min = 0.0
        f_sp = 200.0 / 3
        mels = (frequencies - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        
        if frequencies.ndim > 0:
            log_mask = (frequencies >= min_log_hz)
            mels[log_mask] = min_log_mel + np.log(frequencies[log_mask] / min_log_hz) / logstep
        elif frequencies >= min_log_hz:
            mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
        return mels

    def _mel_to_hz(self, mels):
        mels = np.asanyarray(mels)
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        
        if mels.ndim > 0:
            log_mask = (mels >= min_log_mel)
            freqs[log_mask] = min_log_hz * np.exp(logstep * (mels[log_mask] - min_log_mel))
        elif mels >= min_log_mel:
            freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
        return freqs

    def _build_mel_basis(self):
        fft_freqs = np.linspace(0, self.sr / 2, int(1 + self.n_fft // 2))
        mel_freqs = self._mel_to_hz(np.linspace(self._hz_to_mel(self.fmin), self._hz_to_mel(self.fmax), self.n_mels + 2))
        
        mel_basis = np.zeros((self.n_mels, len(fft_freqs)))
        for i in range(self.n_mels):
            lower = mel_freqs[i]
            center = mel_freqs[i+1]
            upper = mel_freqs[i+2]
            
            forward = (fft_freqs - lower) / (center - lower)
            backward = (upper - fft_freqs) / (upper - center)
            mel_basis[i] = np.maximum(0, np.minimum(forward, backward))
        
        # Slaney-style 面积归一化
        enorm = 2.0 / (mel_freqs[2:self.n_mels+2] - mel_freqs[:self.n_mels])
        mel_basis *= enorm[:, np.newaxis]
        return mel_basis.astype(np.float32)

    def extract(self, wav: np.ndarray) -> np.ndarray:
        """
        提取 Log-Mel 谱图。
        输入: wav [samples] (float32)
        输出: log_mel [T, 128]
        """
        # 1. 手动 Padding (reflect)
        # 对齐官方: (n_fft - hop_size) // 2 = 384
        padding = (self.n_fft - self.hop_length) // 2
        wav_padded = np.pad(wav, (padding, padding), mode='reflect')
        
        # 2. STFT
        num_frames = 1 + (len(wav_padded) - self.n_fft) // self.hop_length
        frames = np.lib.stride_tricks.as_strided(
            wav_padded, 
            shape=(num_frames, self.n_fft), 
            strides=(wav_padded.strides[0] * self.hop_length, wav_padded.strides[0])
        )
        
        # 执行 RFFT
        stft_res = rfft(frames * self.window, n=self.n_fft, axis=1)
        
        # 3. 计算幅度
        # magnitudes = sqrt(abs(stft)^2 + 1e-9)
        magnitudes = np.sqrt(np.abs(stft_res)**2 + 1e-9).T
        
        # 4. Mel 映射
        mel_spec = np.dot(self.mel_basis, magnitudes)
        
        # 5. 动态范围压缩 (Log-Mel)
        log_mel = np.log(np.maximum(mel_spec, 1e-5))
        
        return log_mel.T
