import numpy as np
import pandas as pd

def calculate_rsi(prices, period=14):
    if len(prices) < period: return np.zeros_like(prices)
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0: rs = 100
    else: rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        upval = delta if delta > 0 else 0.
        downval = -delta if delta < 0 else 0.
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        if down == 0: rs = 100
        else: rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def calculate_ema(prices, period=20):
    if len(prices) < period: return np.array(prices)
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    if len(prices) < period: return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
    sma = pd.Series(prices).rolling(window=period).mean()
    std = pd.Series(prices).rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band.values, sma.values, lower_band.values

def calculate_macd(prices, slow=26, fast=12, signal=9):
    if len(prices) < slow: return np.zeros_like(prices), np.zeros_like(prices)
    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.values, signal_line.values

def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period: return np.zeros_like(closes)
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    atr = pd.Series(tr).rolling(window=period).mean()
    return np.concatenate([[0], atr.values])

def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
    if len(closes) < k_period: return np.zeros_like(closes), np.zeros_like(closes)
    low_min = pd.Series(lows).rolling(window=k_period).min()
    high_max = pd.Series(highs).rolling(window=k_period).max()
    k = 100 * (pd.Series(closes) - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k.values, d.values

def calculate_adx(highs, lows, closes, period=14):
    if len(closes) < period * 2: return np.zeros_like(closes)
    # Simplified ADX implementation
    plus_dm = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]), np.maximum(highs[1:] - highs[:-1], 0), 0)
    minus_dm = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]), np.maximum(lows[:-1] - lows[1:], 0), 0)

    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))

    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return np.concatenate([[0], adx.values])
