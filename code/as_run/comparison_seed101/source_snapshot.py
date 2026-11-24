# -*- coding: utf-8 -*-
"""SDWPF 多步风电功率预测与对比实验。

主模型结合 TCN、动态尾流启发图、传播延迟、软异构角色门控和自适应图；
可通过 --experiments 选择消融与对比模型。输出各步预测、指标和训练记录。
默认数据路径指向当前用户桌面的“project_root/SDWPF”，也可由命令行覆盖。

注意：当前数据处理会前向填充功率并将其用作预测标签。正式评估原始
观测标签时，必须另行保留有效标签掩码并重训，不能将本脚本的结果视为
已完成该修正的实验。
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# 1. 配置
# =========================

@dataclass
class Config:
    data_csv: str = r"C:\Users\Administrator\Desktop\project_root\SDWPF\sdwpf_2021_full.csv"
    location_csv: str = r"C:\Users\Administrator\Desktop\project_root\SDWPF\sdwpf_turb_location_elevation.csv"
    out_dir: str = "sdwpf_soft_hetero_wake_gat_results"

    # 0 表示使用全部风机。
    max_turbines: int = 0
    rated_power_kw: float = 1500.0
    pred_horizon: int = 6 # 输出 1..pred_horizon 的多步预测结果

    # 数据切分：按时间顺序切分，避免未来信息泄露。
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # 三种尾流启发交互分数的固定参数。
    rotor_D: float = 82.0
    ct: float = 0.8
    niayifar_k: float = 0.06
    niayifar_alpha: float = 0.12
    bpa_ky: float = 0.035
    jimenez_kd: float = 0.05
    w_niayifar: float = 0.6
    w_bpa: float = 2.0
    w_jimenez: float = 0.6

    # 动态图的边筛选阈值。
    wake_edge_threshold: float = 0.08

    # 轻量版 TCN-Wake-GAT 训练参数。
    hidden_dim: int = 96
    num_layers: int = 2
    dropout: float = 0.15
    gat_heads: int = 2
    gat_top_k: int = 8

    # 轻量 TCN 时间编码参数。temporal_window=1 时退化为仅使用当前时刻。
    temporal_window: int = 3
    tcn_layers: int = 1
    tcn_kernel_size: int = 3

    # 物理图注意力中的软角色门控强度。
    hetero_gate_alpha: float = 1.5

    # 自适应图残差分支；top_k=0 使用稠密图。
    adaptive_graph_alpha: float = 0.3
    adaptive_graph_emb_dim: int = 32
    adaptive_graph_top_k: int =24

    # 依据下风距离与风速估计延迟，并在历史窗口内插值取上游状态。
    use_wake_delay: int = 1
    sample_interval_minutes: float = 10.0
    wake_delay_speed_floor: float = 3.0
    wake_delay_max_steps: int = 2

    # 物理图与自适应图之间的可学习门控；0 表示固定残差融合。
    use_learnable_fusion: int = 1

    epochs: int = 50
    batch_size: int = 4096
    lr: float = 5e-4
    weight_decay: float = 1e-5
    # Farm-level auxiliary loss weight.
    # loss = node_loss + farm_loss_weight * farm_loss
    # farm_loss 在归一化功率空间中按时间步对所有风机取均值，避免全场求和导致损失尺度过大。
    farm_loss_weight: float = 0.13
    patience: int = 8
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 主模型与当前选用的对比模型。
    experiments: str = "full,lstm,cnn_lstm,cnn_bilstm,transformer,dbfp_net,stgcn,agcrn,stv_pgnn_adapted"


# =========================
# 2. 基础工具函数
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def circular_mean_deg(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    rad = np.deg2rad(np.mod(values, 360.0))
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    return float((np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """返回 a-b 的最小角度差，范围 [-180,180)。"""
    return ((a - b + 180.0) % 360.0) - 180.0


def compute_metrics(
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
    floor_kw: float = 100.0,
    capacity_kw: float = None,
) -> Dict[str, float]:
    """
    Deterministic wind-power forecasting metrics.

    Returned metrics:
    - rmse, mae, r2: standard absolute-error and goodness-of-fit metrics.
    - mape: only computed where |true| >= floor_kw to avoid near-zero explosion.
    - smape: symmetric MAPE, more stable than MAPE near low-power periods.
    - wape: weighted absolute percentage error, sum(|e|)/sum(|y|).
    - mbe: mean bias error = mean(pred - true), indicating over/under prediction.
    - nrmse, nmae, nbias: capacity-normalized percentages when capacity_kw is given.
    - corr: Pearson correlation coefficient between prediction and observation.
    - maxae, p95ae: maximum and 95th-percentile absolute error, useful for extreme errors.
    """
    yt = np.asarray(y_true_kw, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred_kw, dtype=np.float64).reshape(-1)
    ok = np.isfinite(yt) & np.isfinite(yp)
    if ok.sum() == 0:
        return {
            "rmse": np.nan, "mae": np.nan, "mape": np.nan, "smape": np.nan, "wape": np.nan,
            "mbe": np.nan, "nrmse": np.nan, "nmae": np.nan, "nbias": np.nan,
            "r2": np.nan, "corr": np.nan, "maxae": np.nan, "p95ae": np.nan,
        }

    yt = yt[ok]
    yp = yp[ok]
    err_true_minus_pred = yt - yp
    err_pred_minus_true = yp - yt
    abs_err = np.abs(err_true_minus_pred)

    rmse = float(np.sqrt(np.mean(err_true_minus_pred ** 2)))
    mae = float(np.mean(abs_err))

    # MAPE is not reliable when true power is close to zero, so use floor_kw.
    mask = np.abs(yt) >= float(floor_kw)
    mape = float(np.mean(abs_err[mask] / np.maximum(np.abs(yt[mask]), 1.0)) * 100.0) if mask.any() else np.nan

    smape = float(np.mean(2.0 * abs_err / (np.abs(yt) + np.abs(yp) + 1e-8)) * 100.0)
    wape = float(np.sum(abs_err) / (np.sum(np.abs(yt)) + 1e-8) * 100.0)

    # Bias: positive means overprediction on average; negative means underprediction.
    mbe = float(np.mean(err_pred_minus_true))

    r2 = float(1.0 - np.sum(err_true_minus_pred ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-8))
    if yt.size >= 2 and np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        corr = np.nan

    maxae = float(np.max(abs_err))
    p95ae = float(np.percentile(abs_err, 95))

    if capacity_kw is not None and float(capacity_kw) > 0:
        cap = float(capacity_kw)
        nrmse = float(rmse / cap * 100.0)
        nmae = float(mae / cap * 100.0)
        nbias = float(mbe / cap * 100.0)
    else:
        nrmse = np.nan
        nmae = np.nan
        nbias = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "mbe": mbe,
        "nrmse": nrmse,
        "nmae": nmae,
        "nbias": nbias,
        "r2": r2,
        "corr": corr,
        "maxae": maxae,
        "p95ae": p95ae,
    }


# =========================
# 3. SDWPF 数据读取与布局读取
# =========================

def _resolve_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"缺少必要列之一: {candidates}. 当前列名: {list(df.columns)[:40]}")
    return ""


def load_sdwpf_long(cfg: Config) -> pd.DataFrame:
    if not os.path.isfile(cfg.data_csv):
        raise FileNotFoundError(f"找不到数据文件: {cfg.data_csv}")

    print(f"[Data] 读取 SDWPF 数据: {cfg.data_csv}")
    header = pd.read_csv(cfg.data_csv, nrows=0)
    cols = list(header.columns)
    lower_to_col = {str(c).strip().lower(): c for c in cols}

    need_lower = {
        "turbid", "tmstamp", "timestamp", "date and time", "time", "day",
        "wspd", "wdir", "ndir", "patv",
    }
    usecols = [orig for low, orig in lower_to_col.items() if low in need_lower]
    if not usecols:
        raise ValueError("未识别到 SDWPF 必要列，请检查 TurbID/Tmstamp/Wspd/Wdir/Ndir/Patv 是否存在。")

    df = pd.read_csv(cfg.data_csv, usecols=usecols)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    turb_col = _resolve_column(df, ["TurbID", "TurbId", "turb_id"])
    time_col = _resolve_column(df, ["Tmstamp", "Timestamp", "Date and time", "Time"])
    wspd_col = _resolve_column(df, ["Wspd", "Wind speed (m/s)"])
    wdir_col = _resolve_column(df, ["Wdir", "Wind direction (°)"])
    ndir_col = _resolve_column(df, ["Ndir", "Nacelle position (°)"])
    patv_col = _resolve_column(df, ["Patv", "Power (kW)"])
    day_col = _resolve_column(df, ["Day"], required=False)

    for c in [turb_col, wspd_col, wdir_col, patv_col] + ([ndir_col] if ndir_col else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[turb_col]).copy()
    df[turb_col] = df[turb_col].astype(int)

    # 时间解析：兼容完整日期时间，也兼容 Day + HH:MM。
    ts = pd.to_datetime(df[time_col], errors="coerce")
    if ts.isna().all():
        tm = df[time_col].astype(str).str.strip()
        parts = tm.str.split(":", expand=True)
        hh = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
        mm = pd.to_numeric(parts[1] if parts.shape[1] > 1 else 0, errors="coerce").fillna(0).astype(int)
        ss = pd.to_numeric(parts[2] if parts.shape[1] > 2 else 0, errors="coerce").fillna(0).astype(int)
        if day_col:
            day = pd.to_numeric(df[day_col], errors="coerce").fillna(1).astype(int)
        else:
            day = df.groupby(turb_col).cumcount() // 144 + 1
        base = pd.Timestamp("2021-01-01")
        ts = base + pd.to_timedelta(day - 1, unit="D") + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m") + pd.to_timedelta(ss, unit="s")

    out = pd.DataFrame({
        "time": ts,
        "turbine": df[turb_col].map(lambda x: f"TurbID {int(x)}"),
        "turb_id": df[turb_col].astype(int),
        "ws": df[wspd_col].astype(float),
        "wd": np.mod(df[wdir_col].astype(float), 360.0),
        "yaw": np.mod(df[ndir_col].astype(float), 360.0),
        "power": df[patv_col].astype(float),
    })

    # 基本物理清洗：负风速/负功率记为 NaN，后续只用前向填充，不使用未来信息。
    out.loc[out["ws"] < 0, "ws"] = np.nan
    out.loc[out["power"] < 0, "power"] = np.nan
    out = out.dropna(subset=["time", "turbine"]).sort_values(["time", "turb_id"]).reset_index(drop=True)
    print(f"[Data] 读取完成: 记录数={len(out)}, 风机数={out['turbine'].nunique()}, 时间步={out['time'].nunique()}")
    return out


def load_layout(cfg: Config, turbines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(cfg.location_csv):
        raise FileNotFoundError(f"找不到位置文件: {cfg.location_csv}")

    loc = pd.read_csv(cfg.location_csv)
    loc = loc.rename(columns={c: str(c).strip() for c in loc.columns})
    tid_col = _resolve_column(loc, ["TurbID", "TurbId", "turb_id"])

    # 兼容常见列名：x/y、X/Y、Lon/Lat。
    x_col = _resolve_column(loc, ["x", "X", "x(m)", "X(m)", "Lon", "Longitude", "longitude"], required=False)
    y_col = _resolve_column(loc, ["y", "Y", "y(m)", "Y(m)", "Lat", "Latitude", "latitude"], required=False)
    if not x_col or not y_col:
        raise ValueError(f"位置文件需要包含 TurbID 和 x/y 或 Lon/Lat 列。当前列名: {list(loc.columns)}")

    loc[tid_col] = pd.to_numeric(loc[tid_col], errors="coerce")
    loc["turbine"] = loc[tid_col].map(lambda x: f"TurbID {int(x)}" if pd.notna(x) else None)
    loc[x_col] = pd.to_numeric(loc[x_col], errors="coerce")
    loc[y_col] = pd.to_numeric(loc[y_col], errors="coerce")
    mp_x = dict(zip(loc["turbine"], loc[x_col]))
    mp_y = dict(zip(loc["turbine"], loc[y_col]))
    x = np.array([mp_x.get(t, np.nan) for t in turbines], dtype=np.float64)
    y = np.array([mp_y.get(t, np.nan) for t in turbines], dtype=np.float64)

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        missing = [t for t, xi, yi in zip(turbines, x, y) if not np.isfinite(xi) or not np.isfinite(yi)]
        raise ValueError(f"位置文件缺少这些风机坐标: {missing[:10]}")

    # 若是经纬度，转成近似米制坐标；若本来是米制坐标，不影响相对布局。
    if np.nanmax(np.abs(x)) <= 180.0 and np.nanmax(np.abs(y)) <= 90.0:
        lon = x.copy()
        lat = y.copy()
        lat0 = np.deg2rad(np.nanmean(lat))
        x_m = (np.deg2rad(lon) - np.deg2rad(np.nanmean(lon))) * 6371000.0 * np.cos(lat0)
        y_m = (np.deg2rad(lat) - np.deg2rad(np.nanmean(lat))) * 6371000.0
        x, y = x_m, y_m

    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    return x.astype(np.float32), y.astype(np.float32)


def make_wide_matrices(df: pd.DataFrame, turbines: List[str]) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def pivot(value: str) -> pd.DataFrame:
        p = df.pivot_table(index="time", columns="turbine", values=value, aggfunc="mean").reindex(columns=turbines).sort_index()
        return p

    p_ws = pivot("ws")
    p_wd = pivot("wd")
    p_yaw = pivot("yaw")
    p_power = pivot("power")
    times = pd.DatetimeIndex(p_power.index)

    def causal_clean(p: pd.DataFrame, low=None, high=None, fill=0.0) -> np.ndarray:
        arr = p.copy()
        for c in arr.columns:
            s = pd.to_numeric(arr[c], errors="coerce")
            if low is not None:
                s = s.mask(s < low, np.nan)
            if high is not None:
                s = s.mask(s > high, np.nan)
            s = s.ffill().fillna(fill)
            arr[c] = s
        return arr.to_numpy(dtype=np.float32)

    ws = causal_clean(p_ws, low=0.0, high=40.0, fill=0.0)
    wd = np.mod(causal_clean(p_wd, low=0.0, high=360.0, fill=0.0), 360.0)
    yaw = np.mod(causal_clean(p_yaw, low=0.0, high=360.0, fill=0.0), 360.0)
    power = causal_clean(p_power, low=0.0, high=None, fill=0.0)
    return times, ws, wd, yaw, power


# =========================
# 4. 三物理尾流公式 + 异构动态类型
# =========================

def build_three_wake_matrices(x: np.ndarray, y: np.ndarray, wd_deg: float, yaw_misalignment_deg: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """
    构建三种物理尾流矩阵 W[src, dst]：
    - niayifar: Jensen/Niayifar-like top-hat + Gaussian radial decay
    - bpa: Bastankhah & Porte-Agel Gaussian wake
    - jimenez: Jimenez yaw deflection + Gaussian wake
    - mix: 三者加权融合
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    yaw_misalignment = np.asarray(yaw_misalignment_deg, dtype=np.float64)
    n = x.size

    flow = np.array([math.cos(math.radians(wd_deg)), math.sin(math.radians(wd_deg))], dtype=np.float64)
    dx = x[None, :] - x[:, None]   # src row -> dst col
    dy = y[None, :] - y[:, None]
    axial = dx * flow[0] + dy * flow[1]
    # Signed crosswind coordinate: retain its sign until yaw displacement is applied.
    crosswind = -dx * flow[1] + dy * flow[0]
    radial = np.abs(crosswind)
    mask = axial > 0.0
    np.fill_diagonal(mask, False)

    ct = float(cfg.ct)
    D = float(cfg.rotor_D)
    yaw_rel = np.deg2rad(yaw_misalignment)[:, None]
    wake_align = np.abs(np.cos(yaw_rel))

    # (1) Niayifar/Jensen-like component
    spread = D * (1.0 + float(cfg.niayifar_alpha) * np.maximum(axial, 0.0) / D)
    spread = np.maximum(spread, 1.0)
    radial_decay = np.exp(-0.5 * (radial / spread) ** 2)
    stream_decay = math.sqrt(ct) / (1.0 + float(cfg.niayifar_k) * np.maximum(axial, 0.0) / D)
    w_ni = stream_decay * wake_align * radial_decay

    # (2) BPA Gaussian component
    ct_eff = ct * (wake_align ** 2)
    sigma = float(cfg.bpa_ky) * np.maximum(axial, 0.0) + D / math.sqrt(8.0)
    sigma = np.maximum(sigma, 1.0)
    term = ct_eff * (D ** 2) / (8.0 * (sigma ** 2))
    term = np.clip(term, 0.0, 0.999)
    c0 = 1.0 - np.sqrt(1.0 - term)
    w_bpa = c0 * np.exp(-0.5 * (radial / sigma) ** 2)

    # (3) Jimenez yaw deflection component
    delta = float(cfg.jimenez_kd) * np.maximum(axial, 0.0) * np.tan(yaw_rel)
    radial_eff = np.abs(crosswind - delta)
    w_jim = c0 * np.exp(-0.5 * (radial_eff / sigma) ** 2)

    for w in (w_ni, w_bpa, w_jim):
        w[~mask] = 0.0
        np.fill_diagonal(w, 0.0)

    a = max(0.0, float(cfg.w_niayifar))
    b = max(0.0, float(cfg.w_bpa))
    c = max(0.0, float(cfg.w_jimenez))
    s = a + b + c
    if s <= 0.0:
        a = b = c = 1.0
        s = 3.0
    w_mix = (a * w_ni + b * w_bpa + c * w_jim) / s

    return {
        "niayifar": np.clip(w_ni, 0.0, 0.95).astype(np.float32),
        "bpa": np.clip(w_bpa, 0.0, 0.95).astype(np.float32),
        "jimenez": np.clip(w_jim, 0.0, 0.95).astype(np.float32),
        "mix": np.clip(w_mix, 0.0, 0.95).astype(np.float32),
        # Geometry terms used by the delay-aware edge features.
        # axial/radial are only meaningful for downstream pairs; invalid pairs are set to zero.
        "axial": np.where(mask, np.maximum(axial, 0.0), 0.0).astype(np.float32),
        "radial": np.where(mask, radial, 0.0).astype(np.float32),
    }



# =========================
# 5. 轻量版 Wake-GAT 特征与图构建
# =========================

def build_local_node_features(
    ws_t: np.ndarray,
    wd_t: np.ndarray,
    yaw_t: np.ndarray,
    power_t: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """每台风机的低维局部节点特征 [N,6]。"""
    rated = max(float(cfg.rated_power_kw), 1e-6)
    ws_norm = np.clip(ws_t, 0.0, 40.0) / 20.0
    p_norm = np.clip(power_t / rated, 0.0, 1.5)
    absolute_bearing = np.mod(yaw_t + wd_t, 360.0)
    wd_rad = np.deg2rad(absolute_bearing)
    mis = np.deg2rad(angle_diff_deg(yaw_t, absolute_bearing))
    return np.concatenate([
        ws_norm[:, None],
        p_norm[:, None],
        np.sin(wd_rad)[:, None],
        np.cos(wd_rad)[:, None],
        np.sin(mis)[:, None],
        np.cos(mis)[:, None],
    ], axis=1).astype(np.float32)


def build_soft_roles(w_mix: np.ndarray) -> np.ndarray:
    """软异构角色 [N,4]：downstream_role, upstream_role, role_balance, role_strength。"""
    in_mix = w_mix.sum(axis=0)
    out_mix = w_mix.sum(axis=1)
    total = in_mix + out_mix + 1e-6
    downstream_role = in_mix / total       # 受上游尾流影响的程度
    upstream_role = out_mix / total        # 影响下游风机的程度
    role_balance = (out_mix - in_mix) / total
    role_strength = np.maximum(in_mix, out_mix)
    return np.stack([downstream_role, upstream_role, role_balance, role_strength], axis=1).astype(np.float32)


def build_topk_wake_neighbors(w: Dict[str, np.ndarray], cfg: Config, farm_ws: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    为每个 dst 风机构建 top-k 上游邻居。

    edge_attr: [N,K,7]
    0: Niayifar/Jensen-like wake weight
    1: BPA wake weight
    2: Jimenez wake weight
    3: fused wake weight
    4: normalized downstream distance
    5: normalized lateral offset
    6: normalized wake propagation delay
    """
    w_ni = w["niayifar"]
    w_bpa = w["bpa"]
    w_jim = w["jimenez"]
    w_mix = w["mix"]
    axial = w.get("axial", np.zeros_like(w_mix, dtype=np.float32)).astype(np.float32)
    radial = w.get("radial", np.zeros_like(w_mix, dtype=np.float32)).astype(np.float32)

    n = int(w_mix.shape[0])
    k = int(max(1, cfg.gat_top_k))
    thr = float(cfg.wake_edge_threshold)
    D = max(float(cfg.rotor_D), 1e-6)

    # Estimate wake transport delay in sampling steps:
    #   delay_steps = downstream_distance / wind_speed / sample_interval.
    # This is causal: only the current wind speed and layout are used to decide
    # which historical upstream hidden state contributes to the current target.
    if farm_ws is None or not np.isfinite(farm_ws):
        farm_ws = float(getattr(cfg, "wake_delay_speed_floor", 3.0))
    v_wake = max(float(farm_ws), float(getattr(cfg, "wake_delay_speed_floor", 3.0)))
    dt_sec = max(float(getattr(cfg, "sample_interval_minutes", 10.0)) * 60.0, 1.0)
    max_delay_steps = max(1.0, float(getattr(cfg, "wake_delay_max_steps", 3)))
    delay_steps = np.clip(axial / (v_wake * dt_sec + 1e-6), 0.0, max_delay_steps)
    delay_norm = delay_steps / max_delay_steps

    axial_norm = np.clip(axial / (10.0 * D), 0.0, 1.5) / 1.5
    radial_norm = np.clip(radial / (3.0 * D), 0.0, 1.5) / 1.5

    nbr_idx = np.zeros((n, k), dtype=np.int16)
    edge_attr = np.zeros((n, k, 7), dtype=np.float16)
    edge_mask = np.zeros((n, k), dtype=np.bool_)

    for dst in range(n):
        src = np.where(w_mix[:, dst] >= thr)[0]
        if src.size > 0:
            order = np.argsort(-w_mix[src, dst])
            src = src[order[:k]]
            m = int(src.size)
            nbr_idx[dst, :m] = src.astype(np.int16)
            edge_attr[dst, :m, 0] = w_ni[src, dst].astype(np.float16)
            edge_attr[dst, :m, 1] = w_bpa[src, dst].astype(np.float16)
            edge_attr[dst, :m, 2] = w_jim[src, dst].astype(np.float16)
            edge_attr[dst, :m, 3] = w_mix[src, dst].astype(np.float16)
            edge_attr[dst, :m, 4] = axial_norm[src, dst].astype(np.float16)
            edge_attr[dst, :m, 5] = radial_norm[src, dst].astype(np.float16)
            edge_attr[dst, :m, 6] = delay_norm[src, dst].astype(np.float16)
            edge_mask[dst, :m] = True
            if m < k:
                nbr_idx[dst, m:] = dst
        else:
            # 无上游邻居时放一个 dummy self，edge_attr 全 0，保证 softmax 不出现全 mask。
            nbr_idx[dst, 0] = dst
            edge_mask[dst, 0] = True
            if k > 1:
                nbr_idx[dst, 1:] = dst

    soft_roles = build_soft_roles(w_mix)
    return nbr_idx, edge_attr, edge_mask, soft_roles

def build_time_graph_sample(
    ws_t: np.ndarray,
    wd_t: np.ndarray,
    yaw_t: np.ndarray,
    power_t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造单个时间步的节点特征和动态图邻接信息。"""
    x_node = build_local_node_features(ws_t, wd_t, yaw_t, power_t, cfg)  # [N,6]
    n = x_node.shape[0]
    k = int(max(1, cfg.gat_top_k))

    # 只有 full/full_no_adapt/no_hetero_adapt/no_hetero/stgcn 使用物理动态尾流图；
    # LSTM/CNN-LSTM/Transformer/DBFP-Net/CNN-BiLSTM/AGCRN/MTGNN 等基线不使用本文三尾流图，避免把 proposed 的物理图先验泄露给普通时序基线。
    graph_modes = {"full", "full_no_adapt", "no_hetero_adapt", "no_hetero", "stgcn"}
    if mode not in graph_modes:
        nbr_idx = np.zeros((n, k), dtype=np.int16)
        edge_attr = np.zeros((n, k, 7), dtype=np.float16)
        edge_mask = np.zeros((n, k), dtype=np.bool_)
        role = np.zeros((n, 4), dtype=np.float32)
        return x_node, nbr_idx, edge_attr, edge_mask, role

    # Wdir is relative to Ndir; first convert each turbine to an absolute
    # clockwise-from-north wind-from bearing. x points east and y points north.
    absolute_bearing = np.mod(yaw_t + wd_t, 360.0)
    farm_from_bearing = circular_mean_deg(absolute_bearing)
    farm_wd = (270.0 - farm_from_bearing) % 360.0  # downwind mathematical angle
    yaw_misalignment = angle_diff_deg(yaw_t, absolute_bearing)
    w = build_three_wake_matrices(x, y, farm_wd, yaw_misalignment, cfg)
    farm_ws = float(np.nanmean(ws_t[np.isfinite(ws_t)])) if np.isfinite(ws_t).any() else float(getattr(cfg, "wake_delay_speed_floor", 3.0))
    nbr_idx, edge_attr, edge_mask, role = build_topk_wake_neighbors(w, cfg, farm_ws=farm_ws)
    return x_node, nbr_idx, edge_attr, edge_mask, role


def build_dataset(cfg: Config, mode: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], pd.DatetimeIndex]:
    df = load_sdwpf_long(cfg)
    turbines = sorted(df["turbine"].unique().tolist(), key=lambda s: int(str(s).split()[-1]))
    if int(cfg.max_turbines) > 0:
        turbines = turbines[: int(cfg.max_turbines)]
    print(f"[Data] 使用风机数: {len(turbines)} | {turbines[:5]}{' ...' if len(turbines) > 5 else ''}")

    x, y = load_layout(cfg, turbines)
    times, ws, wd, yaw, power = make_wide_matrices(df[df["turbine"].isin(turbines)].copy(), turbines)

    # pred_horizon=H 表示一次性输出 t+1, t+2, ..., t+H，而不是只输出 t+H。
    H = int(max(1, cfg.pred_horizon))
    L = int(max(1, cfg.temporal_window))
    current_start = L - 1
    current_end = power.shape[0] - H  # inclusive current index is current_end - 1
    total_samples = current_end - current_start
    if total_samples <= 10:
        raise RuntimeError("有效时间步太少，无法训练。请减小 temporal_window 或 pred_horizon。")

    print(
        f"[Feature] 构建 TCN-Wake-GAT 多步数据: mode={mode}, 样本数={total_samples}, "
        f"历史窗口={L}, 输出步数=1..{H}, 风机数={len(turbines)}, top_k={int(cfg.gat_top_k)}"
    )
    X_list, Y_list, T_list = [], [], []
    NBR_list, EDGE_list, MASK_list, ROLE_list = [], [], [], []

    for sample_i, t in enumerate(range(current_start, current_end)):
        # 历史窗口局部特征：[L,N,6]，只使用 t-L+1 到 t 的历史信息，不引入未来信息。
        seq_feats = []
        for tt in range(t - L + 1, t + 1):
            seq_feats.append(build_local_node_features(ws[tt], wd[tt], yaw[tt], power[tt], cfg))
        X_seq = np.stack(seq_feats, axis=0).astype(np.float32)

        # 动态尾流图使用当前时刻 t 的风向、偏航和布局构建。
        _, nbr_t, edge_t, mask_t, role_t = build_time_graph_sample(
            ws[t], wd[t], yaw[t], power[t], x, y, cfg, mode=mode
        )

        # 多步标签：[N,H]，依次为 t+1, t+2, ..., t+H。
        y_steps = []
        time_steps = []
        for hh in range(1, H + 1):
            y_steps.append(np.clip(power[t + hh] / max(float(cfg.rated_power_kw), 1e-6), 0.0, 1.5).astype(np.float32))
            time_steps.append(times[t + hh])
        y_t = np.stack(y_steps, axis=1).astype(np.float32)  # [N,H]

        X_list.append(X_seq)
        Y_list.append(y_t)
        NBR_list.append(nbr_t)
        EDGE_list.append(edge_t)
        MASK_list.append(mask_t)
        ROLE_list.append(role_t)
        T_list.append(time_steps)
        if (sample_i + 1) % 2000 == 0 or (sample_i + 1) == total_samples:
            print(f"[Feature] {sample_i + 1}/{total_samples}", flush=True)

    X = np.stack(X_list, axis=0).astype(np.float32)       # [S,L,N,6]
    Y = np.stack(Y_list, axis=0).astype(np.float32)       # [S,N,H]
    NBR = np.stack(NBR_list, axis=0).astype(np.int16)     # [S,N,K]
    EDGE = np.stack(EDGE_list, axis=0).astype(np.float16) # [S,N,K,7]
    MASK = np.stack(MASK_list, axis=0).astype(np.bool_)   # [S,N,K]
    ROLE = np.stack(ROLE_list, axis=0).astype(np.float32) # [S,N,4]
    target_times = np.array(T_list, dtype="datetime64[ns]")  # [S,H]

    n_train = int(total_samples * float(cfg.train_ratio))
    n_val = int(total_samples * (float(cfg.train_ratio) + float(cfg.val_ratio)))
    n_train = max(n_train, 1)
    n_val = max(n_val, n_train + 1)
    n_val = min(n_val, total_samples - 1)

    # 节点历史特征标准化只使用训练集统计量；边权和角色保持物理尺度。
    train_flat = X[:n_train].reshape(-1, X.shape[-1])
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True) + 1e-6
    if mode == "stv_pgnn_adapted":
        cfg._stv_context = build_stv_context(cfg, turbines, x, y, mean, std)
    Xn = (X - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)

    def pack(a: int, b: int) -> Dict[str, np.ndarray]:
        return {
            "X_time": Xn[a:b].astype(np.float32),       # [S,L,N,F]
            "Y_time": Y[a:b].astype(np.float32),        # [S,N,H]
            "nbr_idx": NBR[a:b],
            "edge_attr": EDGE[a:b],
            "edge_mask": MASK[a:b],
            "role": ROLE[a:b].astype(np.float32),
            "Y_matrix": Y[a:b].astype(np.float32),      # [S,N,H]
            "times": target_times[a:b],                 # [S,H]
        }

    train = pack(0, n_train)
    val = pack(n_train, n_val)
    test = pack(n_val, total_samples)
    meta = {
        "feature_dim": X.shape[-1],
        "n_turbines": len(turbines),
        "temporal_window": L,
        "pred_horizon": H,
        "n_train_t": n_train,
        "n_val_t": n_val - n_train,
        "n_test_t": total_samples - n_val,
    }
    print(
        f"[Split] train/val/test 样本数 = {meta['n_train_t']}/{meta['n_val_t']}/{meta['n_test_t']}, "
        f"temporal_window={L}, pred_horizon=1..{H}, node_feature_dim={meta['feature_dim']}"
    )
    # 返回最后一步的时间索引仅为兼容原函数签名；真正预测时间存放在各 split 的 times: [S,H] 中。
    pred_times = pd.DatetimeIndex(target_times[:, -1])
    return train, val, test, turbines, pred_times


# =========================
# 6. 轻量版 Wake-GAT 模型
# =========================

class CausalConvBlock(nn.Module):
    """轻量因果 TCN block：只使用历史窗口内的信息。"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.padding = int((kernel_size - 1) * dilation)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=self.padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=self.padding)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(channels)

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            return x[:, :, :-self.padding]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, C, L]
        residual = x
        y = self._chomp(self.conv1(x))
        y = self.drop(self.act(y))
        y = self._chomp(self.conv2(y))
        y = self.drop(self.act(y))
        y = self.norm(y + residual)
        return y


class TemporalTCNEncoder(nn.Module):
    """
    共享 TCN 时间编码器。

    输入 x_seq: [B,L,N,F]
    输出 h:     [B,N,H]
    """
    def __init__(self, in_dim: int, hidden_dim: int = 96, layers: int = 2, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        blocks = []
        for i in range(max(1, int(layers))):
            blocks.append(
                CausalConvBlock(
                    channels=hidden_dim,
                    kernel_size=int(kernel_size),
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B,L,N,F] -> [B,L,N,H]
        B, L, N, F = x_seq.shape
        x = x_seq.permute(0, 2, 3, 1).reshape(B * N, F, L)  # [B*N,F,L]
        h = self.input_proj(x)                               # [B*N,H,L]
        h = self.blocks(h)                                   # [B*N,H,L]
        h_seq = h.reshape(B, N, -1, L).permute(0, 3, 1, 2)   # [B,L,N,H]
        return self.out_norm(h_seq)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B,L,N,F]
        return self.forward_sequence(x_seq)[:, -1, :, :]


class LocalTCNMLP(nn.Module):
    """no_graph 基线：TCN 时间编码 + 本机 MLP，不使用动态图。"""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
        tcn_layers: int = 2,
        tcn_kernel_size: int = 3,
        out_steps: int = 1,
    ):
        super().__init__()
        self.out_steps = int(max(1, out_steps))
        self.tcn = TemporalTCNEncoder(in_dim, hidden_dim, tcn_layers, tcn_kernel_size, dropout)
        layers = []
        last = hidden_dim
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = hidden_dim
        layers.append(nn.Linear(last, self.out_steps))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, nbr_idx=None, edge_attr=None, edge_mask=None, role=None) -> torch.Tensor:
        # x: [B,L,N,F]
        h = self.tcn(x)  # [B,N,H]
        B, N, Hdim = h.shape
        out = self.net(h.reshape(B * N, Hdim)).reshape(B, N, self.out_steps)
        return torch.sigmoid(out) * 1.5


class LightweightWakeGAT(nn.Module):
    """
    TCN + 轻量版物理边权 Wake-GAT。

    no_hetero：TCN 时间编码 + 三尾流物理边权 attention bias；
    full_no_adapt：TCN + 三尾流物理边权 Wake-GAT + 软异构角色门控；
    full：在 full_no_adapt 基础上进一步加入自适应图残差分支。
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        heads: int = 2,
        dropout: float = 0.2,
        use_soft_hetero: bool = False,
        edge_dim: int = 7,
        hetero_gate_alpha: float = 1.0,
        tcn_layers: int = 2,
        tcn_kernel_size: int = 3,
        out_steps: int = 1,
        use_adaptive_graph: bool = False,
        n_nodes: int = None,
        adaptive_graph_emb_dim: int = 16,
        adaptive_graph_alpha: float = 0.25,
        adaptive_graph_top_k: int = 16,
        use_wake_delay: bool = True,
        wake_delay_max_steps: int = 3,
        use_learnable_fusion: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(max(1, heads))
        self.head_dim = max(1, self.hidden_dim // self.heads)
        self.inner_dim = self.heads * self.head_dim
        self.use_soft_hetero = bool(use_soft_hetero)
        self.hetero_gate_alpha = float(hetero_gate_alpha)
        self.out_steps = int(max(1, out_steps))

        # Whether to add the learnable adaptive graph branch. The proposed full model and the adaptive ablation variants can set this to True; ordinary baselines remain unchanged.
        self.use_adaptive_graph = bool(use_adaptive_graph)
        self.n_nodes = int(n_nodes) if n_nodes is not None else 0
        self.adaptive_graph_alpha = float(adaptive_graph_alpha)
        self.adaptive_graph_top_k = int(adaptive_graph_top_k)
        self.use_wake_delay = bool(use_wake_delay)
        self.wake_delay_max_steps = max(1, int(wake_delay_max_steps))
        self.use_learnable_fusion = bool(use_learnable_fusion)

        self.tcn = TemporalTCNEncoder(in_dim, hidden_dim, tcn_layers, tcn_kernel_size, dropout)
        self.q_proj = nn.Linear(self.hidden_dim, self.inner_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.inner_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.inner_dim)
        self.edge_bias = nn.Linear(edge_dim, self.heads)
        self.out_proj = nn.Linear(self.inner_dim, self.hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        if self.use_adaptive_graph:
            if self.n_nodes <= 0:
                raise ValueError("use_adaptive_graph=True 时必须传入有效的 n_nodes。")
            emb_dim = int(max(1, adaptive_graph_emb_dim))
            self.adaptive_node_emb1 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
            self.adaptive_node_emb2 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
            self.adaptive_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.adaptive_norm = nn.LayerNorm(self.hidden_dim)
            self.adaptive_dropout = nn.Dropout(dropout)
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2 + 4, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden_dim, self.out_steps)

    @staticmethod
    def _gather_nodes(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """x: [B,N,...], idx: [B,N,K] -> [B,N,K,...]"""
        B, N = x.shape[:2]
        tail = x.shape[2:]
        offset = (torch.arange(B, device=x.device, dtype=idx.dtype) * N).view(B, 1, 1)
        flat_idx = (idx + offset).reshape(-1)
        x_flat = x.reshape(B * N, *tail)
        return x_flat[flat_idx].reshape(B, N, idx.shape[-1], *tail)

    @staticmethod
    def _gather_temporal_nodes(x_seq: torch.Tensor, node_idx: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        """x_seq: [B,L,N,...], node_idx/time_idx: [B,N,K] -> [B,N,K,...]"""
        B, L, N = x_seq.shape[:3]
        tail = x_seq.shape[3:]
        node_idx = node_idx.long().clamp(min=0, max=max(N - 1, 0))
        time_idx = time_idx.long().clamp(min=0, max=max(L - 1, 0))
        b_idx = torch.arange(B, device=x_seq.device, dtype=torch.long).view(B, 1, 1)
        flat_idx = (b_idx * L * N + time_idx * N + node_idx).reshape(-1)
        x_flat = x_seq.reshape(B * L * N, *tail)
        return x_flat[flat_idx].reshape(B, N, node_idx.shape[-1], *tail)

    def _adaptive_adj(self, device: torch.device, n_nodes: int) -> torch.Tensor:
        """Learnable adaptive adjacency A[src->dst] in row-normalized form."""
        if n_nodes != self.n_nodes:
            raise RuntimeError(f"Adaptive graph 初始化节点数 {self.n_nodes} 与输入节点数 {n_nodes} 不一致。")
        score = torch.relu(self.adaptive_node_emb1 @ self.adaptive_node_emb2.t())  # [N,N]
        score = score.masked_fill(torch.eye(n_nodes, device=score.device, dtype=torch.bool), 0.0)
        k = int(self.adaptive_graph_top_k)
        if 0 < k < n_nodes:
            vals, idx = torch.topk(score, k=k, dim=-1)
            sparse_score = torch.full_like(score, float("-inf"))
            sparse_score.scatter_(1, idx, vals)
            A = torch.softmax(sparse_score, dim=-1)
        else:
            A = torch.softmax(score, dim=-1)
        return A.to(device)

    def forward(
        self,
        x: torch.Tensor,
        nbr_idx: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        role: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B,L,N,F], nbr_idx: [B,N,K], edge_attr: [B,N,K,7], role: [B,N,4]
        B, L, N, _ = x.shape
        nbr_idx = nbr_idx.long().clamp(min=0, max=max(N - 1, 0))
        edge_attr = edge_attr.float()
        edge_mask = edge_mask.bool()
        role = role.float()
        # Disable every role-dependent path, including the learned fusion gate.
        if not self.use_soft_hetero:
            role = torch.zeros_like(role)

        # Keep the whole TCN hidden sequence so that physical messages can be
        # gathered from a delay-compensated upstream hidden state.
        h_seq = self.tcn.forward_sequence(x)                  # [B,L,N,H]
        h0 = h_seq[:, -1, :, :]                               # [B,N,H]
        q = self.q_proj(h0).view(B, N, self.heads, self.head_dim)

        if self.use_wake_delay and L > 1 and edge_attr.shape[-1] >= 7:
            k_seq = self.k_proj(h_seq).view(B, L, N, self.heads, self.head_dim)
            v_seq = self.v_proj(h_seq).view(B, L, N, self.heads, self.head_dim)
            # edge_attr[...,6] is delay_steps / wake_delay_max_steps.
            # Fractional delays are handled by linear interpolation between two
            # adjacent hidden states in the causal historical window.
            lag_float = (edge_attr[..., 6].clamp(min=0.0, max=1.0) * float(self.wake_delay_max_steps)).clamp(max=float(L - 1))
            lag0 = torch.floor(lag_float).long()
            lag1 = torch.ceil(lag_float).long()
            w1 = (lag_float - lag0.float()).unsqueeze(-1).unsqueeze(-1)
            t0 = (L - 1 - lag0).clamp(min=0, max=L - 1)
            t1 = (L - 1 - lag1).clamp(min=0, max=L - 1)
            k0 = self._gather_temporal_nodes(k_seq, nbr_idx, t0)
            k1 = self._gather_temporal_nodes(k_seq, nbr_idx, t1)
            v0 = self._gather_temporal_nodes(v_seq, nbr_idx, t0)
            v1 = self._gather_temporal_nodes(v_seq, nbr_idx, t1)
            k_nb = (1.0 - w1) * k0 + w1 * k1
            v_nb = (1.0 - w1) * v0 + w1 * v1
        else:
            k = self.k_proj(h0).view(B, N, self.heads, self.head_dim)
            v = self.v_proj(h0).view(B, N, self.heads, self.head_dim)
            k_nb = self._gather_nodes(k, nbr_idx)  # [B,N,K,heads,head_dim]
            v_nb = self._gather_nodes(v, nbr_idx)

        q_dst = q.unsqueeze(2)
        scores = (q_dst * k_nb).sum(dim=-1) / math.sqrt(float(self.head_dim))  # [B,N,K,heads]

        # 三物理尾流边权作为 attention bias。
        scores = scores + self.edge_bias(edge_attr)
        edge_mix = edge_attr[..., 3].clamp(min=0.0, max=1.0)
        scores = scores + torch.log(edge_mix.clamp_min(1e-6)).unsqueeze(-1)
        if self.use_wake_delay and edge_attr.shape[-1] >= 7:
            # A mild delay penalty stabilizes very long transport edges while the
            # learnable edge_bias can still recover useful delayed interactions.
            scores = scores - 0.10 * edge_attr[..., 6].clamp(min=0.0, max=1.0).unsqueeze(-1)

        if self.use_soft_hetero:
            # role[...,0] = downstream_role/incoming_ratio, role[...,1] = upstream_role/outgoing_ratio
            src_role = self._gather_nodes(role, nbr_idx)  # [B,N,K,7]
            src_up = src_role[..., 1]
            dst_down = role[..., 0].unsqueeze(2)
            gate = (0.10 + 0.90 * src_up * dst_down).clamp(min=1e-6, max=1.0)
            scores = scores + self.hetero_gate_alpha * torch.log(gate).unsqueeze(-1)

        scores = scores.masked_fill(~edge_mask.unsqueeze(-1), -1e9)
        attn = torch.softmax(scores, dim=2)
        attn = self.dropout(attn)

        # 物理边权同时用于 message scaling。dummy self edge_attr=0 时，消息为0，由残差保留本机信息。
        msg = (attn.unsqueeze(-1) * v_nb * edge_mix.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        msg = msg.reshape(B, N, self.inner_dim)
        h = self.norm1(h0 + self.dropout(self.out_proj(msg)))

        # 自适应图残差分支：学习物理尾流图未覆盖的全局/隐含空间相关性。
        # 该分支只在 full、no_hetero_adapt 和 adaptive_only 中启用；no_hetero、no_graph 和外部基线不使用。
        if self.use_adaptive_graph:
            A_adapt = self._adaptive_adj(x.device, N)               # [N,N]
            h_adapt_agg = torch.einsum("ij,bjh->bih", A_adapt, h0)  # [B,N,H]
            h_adapt = torch.relu(self.adaptive_proj(torch.cat([h0, h_adapt_agg], dim=-1)))
            if self.use_learnable_fusion:
                # Node-wise learnable fusion: the model decides how much adaptive
                # residual information should be injected into each turbine state.
                fusion_gate = self.fusion_gate(torch.cat([h, h_adapt, role], dim=-1))
                h = self.adaptive_norm(h + self.adaptive_dropout(self.adaptive_graph_alpha * fusion_gate * h_adapt))
            else:
                h = self.adaptive_norm(h + self.adaptive_dropout(self.adaptive_graph_alpha * h_adapt))

        h = self.norm2(h + self.dropout(self.ffn(h)))
        out = self.head(h)  # [B,N,out_steps]
        return torch.sigmoid(out) * 1.5


# =========================
# 6.1 常用对比模型：LSTM / CNN-LSTM / Transformer / DBFP-Net / CNN-BiLSTM / STGCN / AGCRN / MTGNN
# =========================

class SharedLSTMBaseline(nn.Module):
    """LSTM baseline：每台风机共享同一个 LSTM 时间编码器，不使用图结构。"""
    def __init__(self, in_dim: int, hidden_dim: int = 96, num_layers: int = 2, dropout: float = 0.15, out_steps: int = 1):
        super().__init__()
        self.out_steps = int(max(1, out_steps))
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=max(1, int(num_layers)),
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_steps),
        )

    def forward(self, x: torch.Tensor, nbr_idx=None, edge_attr=None, edge_mask=None, role=None) -> torch.Tensor:
        # x: [B,L,N,F]
        B, L, N, F = x.shape
        z = x.permute(0, 2, 1, 3).reshape(B * N, L, F)
        y, _ = self.lstm(z)
        h = y[:, -1, :]
        out = self.head(h).reshape(B, N, self.out_steps)
        return torch.sigmoid(out) * 1.5


class CNNLSTMBaseline(nn.Module):
    """CNN-LSTM baseline：一维卷积提取局部时间模式，再接 LSTM。"""
    def __init__(self, in_dim: int, hidden_dim: int = 96, num_layers: int = 1, dropout: float = 0.15, out_steps: int = 1):
        super().__init__()
        self.out_steps = int(max(1, out_steps))
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=max(1, int(num_layers)), batch_first=True, dropout=lstm_dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, self.out_steps))

    def forward(self, x: torch.Tensor, nbr_idx=None, edge_attr=None, edge_mask=None, role=None) -> torch.Tensor:
        B, L, N, F = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B * N, F, L)     # [B*N,F,L]
        z = self.conv(z).transpose(1, 2)                   # [B*N,L,H]
        y, _ = self.lstm(z)
        out = self.head(y[:, -1, :]).reshape(B, N, self.out_steps)
        return torch.sigmoid(out) * 1.5


class CNNBiLSTMBaseline(nn.Module):
    """CNN-BiLSTM baseline：卷积 + 双向 LSTM。"""
    def __init__(self, in_dim: int, hidden_dim: int = 96, num_layers: int = 1, dropout: float = 0.15, out_steps: int = 1):
        super().__init__()
        self.out_steps = int(max(1, out_steps))
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 双向后拼接维度为 hidden_dim，因此单向 hidden_dim 取一半。
        bi_hidden = max(8, hidden_dim // 2)
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.bilstm = nn.LSTM(hidden_dim, bi_hidden, num_layers=max(1, int(num_layers)), batch_first=True, dropout=lstm_dropout, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * bi_hidden), nn.Dropout(dropout), nn.Linear(2 * bi_hidden, self.out_steps))

    def forward(self, x: torch.Tensor, nbr_idx=None, edge_attr=None, edge_mask=None, role=None) -> torch.Tensor:
        B, L, N, F = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B * N, F, L)
        z = self.conv(z).transpose(1, 2)
        y, _ = self.bilstm(z)
        out = self.head(y[:, -1, :]).reshape(B, N, self.out_steps)
        return torch.sigmoid(out) * 1.5


class TemporalTransformerBaseline(nn.Module):
    """Transformer baseline：每台风机共享轻量时间 Transformer，不使用图结构。"""
    def __init__(self, in_dim: int, hidden_dim: int = 96, heads: int = 2, layers: int = 2, dropout: float = 0.15, out_steps: int = 1, max_len: int = 512):
        super().__init__()
        self.out_steps = int(max(1, out_steps))
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(layers)))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, self.out_steps))

    def forward(self, x: torch.Tensor, nbr_idx=None, edge_attr=None, edge_mask=None, role=None) -> torch.Tensor:
        B, L, N, F = x.shape
        z = x.permute(0, 2, 1, 3).reshape(B * N, L, F)
        h = self.input_proj(z)
        if L <= self.pos_emb.shape[1]:
            h = h + self.pos_emb[:, :L, :]
        h = self.encoder(h)
        out = self.head(h[:, -1, :]).reshape(B, N, self.out_steps)
        return torch.sigmoid(out) * 1.5


class LightweightDBFPNetBaseline(nn.Module):
    """
    轻量 DBFP-Net 风格 baseline：
    多尺度时间卷积分支 + 双向 LSTM 时序分支融合。
    注意：这是用于公平对比的轻量实现，并非严格复现某篇 DBFP-Net 的全部细节。
    """
    def __init__(self, in_dim: int, hidden_dim: int = 96, dropout: float = 0.15, out_steps: int = 1):
        super().__init__()
        self.out_steps = int(max(1, out_steps))
        branch_dim = max(16, hidden_dim // 3)
        self.conv2 = nn.Conv1d(in_dim, branch_dim, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_dim, branch_dim, kernel_size=5, padding=2)
        bi_hidden = max(16, hidden_dim // 2)
        self.bilstm = nn.LSTM(in_dim, bi_hidden, batch_first=True, bidirectional=True)
        fuse_dim = branch_dim * 3 + bi_hidden * 2
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_steps),
        )

    @staticmethod
    def _last_valid_conv(conv_out: torch.Tensor, L: int) -> torch.Tensor:
        # conv_out: [B*N,C,L']，取末端并裁剪可能由 padding 造成的额外长度。
        return conv_out[:, :, :L].transpose(1, 2)[:, -1, :]

    def forward(self, x: torch.Tensor, nbr_idx=None, edge_attr=None, edge_mask=None, role=None) -> torch.Tensor:
        B, L, N, F = x.shape
        z_seq = x.permute(0, 2, 1, 3).reshape(B * N, L, F)
        z_conv = z_seq.transpose(1, 2)
        c2 = self._last_valid_conv(torch.relu(self.conv2(z_conv)), L)
        c3 = self._last_valid_conv(torch.relu(self.conv3(z_conv)), L)
        c5 = self._last_valid_conv(torch.relu(self.conv5(z_conv)), L)
        y, _ = self.bilstm(z_seq)
        h = torch.cat([c2, c3, c5, y[:, -1, :]], dim=-1)
        out = self.head(h).reshape(B, N, self.out_steps)
        return torch.sigmoid(out) * 1.5


class GraphWeightedConv(nn.Module):
    """基于 top-k 物理边权的简单图卷积，用于 STGCN baseline。"""
    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.lin_nei = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _gather_nodes(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B, N = x.shape[:2]
        tail = x.shape[2:]
        offset = (torch.arange(B, device=x.device, dtype=idx.dtype) * N).view(B, 1, 1)
        flat_idx = (idx + offset).reshape(-1)
        x_flat = x.reshape(B * N, *tail)
        return x_flat[flat_idx].reshape(B, N, idx.shape[-1], *tail)

    def forward(self, h: torch.Tensor, nbr_idx: torch.Tensor, edge_attr: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        B, N, Hdim = h.shape
        idx = nbr_idx.long().clamp(min=0, max=max(N - 1, 0))
        h_nb = self._gather_nodes(h, idx)                         # [B,N,K,H]
        w = edge_attr.float()[..., 3].clamp(min=0.0)               # [B,N,K]
        w = w.masked_fill(~edge_mask.bool(), 0.0)
        denom = w.sum(dim=2, keepdim=True).clamp_min(1e-6)
        agg = (h_nb * (w / denom).unsqueeze(-1)).sum(dim=2)
        out = self.lin_self(h) + self.lin_nei(agg)
        return self.norm(h + self.drop(torch.relu(out)))


class STGCNBaseline(nn.Module):
    """STGCN baseline：TCN 时间编码 + 物理图加权卷积，不使用 GAT 注意力和软异构门控。"""
    def __init__(self, in_dim: int, hidden_dim: int = 96, dropout: float = 0.15, tcn_layers: int = 1, tcn_kernel_size: int = 3, out_steps: int = 1):
        super().__init__()
        self.out_steps = int(max(1, out_steps))
        self.tcn = TemporalTCNEncoder(in_dim, hidden_dim, tcn_layers, tcn_kernel_size, dropout)
        self.gcn1 = GraphWeightedConv(hidden_dim, dropout)
        self.gcn2 = GraphWeightedConv(hidden_dim, dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, self.out_steps))

    def forward(self, x: torch.Tensor, nbr_idx: torch.Tensor, edge_attr: torch.Tensor, edge_mask: torch.Tensor, role=None) -> torch.Tensor:
        h = self.tcn(x)
        h = self.gcn1(h, nbr_idx, edge_attr, edge_mask)
        h = self.gcn2(h, nbr_idx, edge_attr, edge_mask)
        out = self.head(h)
        return torch.sigmoid(out) * 1.5


class AGCRNBaseline(nn.Module):
    """AGCRN 风格 baseline：自适应图 + GRU 时间递推，不使用物理尾流图。"""
    def __init__(self, in_dim: int, n_nodes: int, hidden_dim: int = 96, dropout: float = 0.15, out_steps: int = 1, emb_dim: int = 16):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.out_steps = int(max(1, out_steps))
        self.node_emb1 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, self.out_steps))

    def _adaptive_adj(self, device) -> torch.Tensor:
        A = torch.relu(self.node_emb1 @ self.node_emb2.t())
        A = torch.softmax(A, dim=-1)
        return A.to(device)

    def forward(self, x: torch.Tensor, nbr_idx=None, edge_attr=None, edge_mask=None, role=None) -> torch.Tensor:
        B, L, N, F = x.shape
        if N != self.n_nodes:
            raise RuntimeError(f"AGCRNBaseline 初始化节点数 {self.n_nodes} 与输入节点数 {N} 不一致。")
        A = self._adaptive_adj(x.device)
        h = torch.zeros(B, N, self.hidden_dim, device=x.device, dtype=x.dtype)
        for tt in range(L):
            inp = torch.relu(self.in_proj(x[:, tt, :, :]))
            agg = torch.einsum("ij,bjh->bih", A, h)
            h = self.gru((inp + agg).reshape(B * N, self.hidden_dim), h.reshape(B * N, self.hidden_dim)).reshape(B, N, self.hidden_dim)
            h = self.drop(h)
        out = self.head(h)
        return torch.sigmoid(out) * 1.5


class MTGNNBaseline(nn.Module):
    """MTGNN 风格 baseline：自适应图学习 + 门控膨胀时间卷积。"""
    def __init__(self, in_dim: int, n_nodes: int, hidden_dim: int = 96, dropout: float = 0.15, out_steps: int = 1, emb_dim: int = 16):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.out_steps = int(max(1, out_steps))
        self.node_emb1 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
        self.start_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.filter_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 1))
        self.gate_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 1))
        self.mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, self.out_steps)

    def _adaptive_adj(self, device) -> torch.Tensor:
        A = torch.relu(self.node_emb1 @ self.node_emb2.t())
        A = torch.softmax(A, dim=-1)
        return A.to(device)

    @staticmethod
    def _chomp_time(x: torch.Tensor, L: int) -> torch.Tensor:
        # x: [B,H,N,Lpad]
        return x[..., :L]

    def forward(self, x: torch.Tensor, nbr_idx=None, edge_attr=None, edge_mask=None, role=None) -> torch.Tensor:
        B, L, N, F = x.shape
        if N != self.n_nodes:
            raise RuntimeError(f"MTGNNBaseline 初始化节点数 {self.n_nodes} 与输入节点数 {N} 不一致。")
        z = x.permute(0, 3, 2, 1)          # [B,F,N,L]
        h = self.start_conv(z)             # [B,H,N,L]
        filt = torch.tanh(self._chomp_time(self.filter_conv(h), L))
        gate = torch.sigmoid(self._chomp_time(self.gate_conv(h), L))
        h = self.drop(filt * gate)
        h_last = h[..., -1].permute(0, 2, 1)  # [B,N,H]
        A = self._adaptive_adj(x.device)
        agg = torch.einsum("ij,bjh->bih", A, h_last)
        h_out = self.norm(torch.relu(self.mix(torch.cat([h_last, agg], dim=-1))))
        out = self.head(h_out)
        return torch.sigmoid(out) * 1.5




# 内置STV-PGNN-adapted物理对比：沿用之前的适配实现，并非论文3D-ADG求解器的严格复现。
def build_stv_context(cfg, turbines, x, y, mean, std):
    """Read extra static elevation only for the new baseline; reuse original x/y."""
    layout=pd.read_csv(cfg.location_csv)
    columns={str(c).strip().lower():c for c in layout.columns}
    tid=next((columns[k] for k in ["turbid","turb_id"] if k in columns),None)
    if tid is None:
        raise ValueError("STV-PGNN-adapted requires turbine IDs in the layout")
    elevation=next((columns[k] for k in ["ele","elevation","height"] if k in columns),None)
    ids=[int(str(t).split()[-1]) for t in turbines]
    if elevation:
        mapping=dict(zip(pd.to_numeric(layout[tid]),pd.to_numeric(layout[elevation])))
        z=np.array([mapping.get(t,np.nan) for t in ids],dtype=np.float32)
        if not np.isfinite(z).all():
            raise ValueError("Missing turbine elevation for STV-PGNN-adapted")
        z=z-z.mean()
    else:
        z=np.zeros(len(ids),dtype=np.float32)
    return dict(xyz=np.column_stack([x,y,z]).astype(np.float32),
                mean=np.asarray(mean,dtype=np.float32).reshape(-1),
                std=np.asarray(std,dtype=np.float32).reshape(-1))


def _stv_mlp(in_dim,hidden,out_dim):
    return nn.Sequential(nn.Linear(in_dim,hidden),nn.LeakyReLU(.1),
        nn.Linear(hidden,hidden),nn.LeakyReLU(.1),nn.Linear(hidden,hidden),
        nn.LeakyReLU(.1),nn.Linear(hidden,out_dim))


class STVPGNNAdaptedBaseline(nn.Module):
    """Five-argument model interface compatible with the original train/evaluate."""
    def __init__(self,cfg,in_dim,n_nodes):
        super().__init__()
        context=getattr(cfg,"_stv_context",None)
        if context is None:
            raise ValueError("Build the original dataset in stv_pgnn_adapted mode before creating this baseline")
        if in_dim!=6 or context["xyz"].shape!=(n_nodes,3):
            raise ValueError("STV baseline expects the original six local features and matching layout")
        self.register_buffer("xyz",torch.as_tensor(context["xyz"]).clone())
        self.register_buffer("feature_mean",torch.as_tensor(context["mean"]).clone())
        self.register_buffer("feature_std",torch.as_tensor(context["std"]).clone())
        self.D=float(cfg.rotor_D)
        self.top_k=max(1,int(cfg.gat_top_k))
        self.history=int(cfg.temporal_window)
        self.raw_physics=nn.Parameter(torch.zeros(5))
        h=int(cfg.hidden_dim)
        self.node_in=nn.Linear(in_dim*self.history,h)
        self.global_in=nn.Linear(3,h)
        self.edge_in=nn.Linear(4,h)
        self.edge_updates=nn.ModuleList([_stv_mlp(4*h,h,h) for _ in range(3)])
        self.node_updates=nn.ModuleList([_stv_mlp(3*h,h,h) for _ in range(3)])
        self.global_updates=nn.ModuleList([_stv_mlp(3*h,h,h) for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
        self.head=_stv_mlp(h,h,int(cfg.pred_horizon))

    def _physics(self,x_last):
        # Recover ONLY this baseline's forecast-origin physical inputs from the
        # original standardized local features. This does not change X or the
        # preprocessing used by Full or any of the seven existing baselines.
        local=x_last*self.feature_std+self.feature_mean
        ws=local[...,0]*20
        relative=torch.atan2(local[...,2],local[...,3])
        original_misalignment=torch.atan2(local[...,4],local[...,5])
        nacelle=relative+original_misalignment
        absolute=nacelle+relative
        psi=torch.atan2(absolute.sin().mean(1),absolute.cos().mean(1))
        theta=1.5*math.pi-psi
        gamma=torch.atan2((nacelle-absolute).sin(),(nacelle-absolute).cos())
        delta=self.xyz[None,:,:]-self.xyz[:,None,:]
        dx,dy,dz=delta[...,0].T,delta[...,1].T,delta[...,2].T
        axial=theta.cos()[:,None,None]*dx+theta.sin()[:,None,None]*dy
        cross=-theta.sin()[:,None,None]*dx+theta.cos()[:,None,None]*dy
        ct=.1+.8*torch.sigmoid(self.raw_physics[0])
        kp=.01+.09*torch.sigmoid(self.raw_physics[1])
        km=.01+.09*torch.sigmoid(self.raw_physics[2])
        shear=.05+.25*torch.sigmoid(self.raw_physics[3])
        turbulence=.04+.16*torch.sigmoid(self.raw_physics[4])
        a=axial.clamp_min(0)
        boost=1+.4*ct/((a/self.D).clamp_min(1)*turbulence)
        sp=self.D/math.sqrt(8)+kp*boost*a
        sm=self.D/math.sqrt(8)+km*boost*a
        sz=(sp*sm).sqrt()
        cross=cross-(-.3*ct*gamma[:,None,:]*a)
        sep=.25*self.D*torch.exp(-a/(4*self.D))
        lobe=.5*torch.exp(-.5*(((cross-sep)/sp)**2+(dz/sz)**2))
        lobe=lobe+.5*torch.exp(-.5*(((cross+sep)/sm)**2+(dz/sz)**2))
        amplitude=1-(1-(ct*self.D**2/(8*sp*sm)).clamp(0,.999)).sqrt()
        height_factor=((70+dz).clamp_min(1)/70)**shear
        weight=(amplitude*lobe/height_factor).clamp(0,.95)
        N=x_last.shape[1]
        valid=(axial>0)&~torch.eye(N,device=x_last.device,dtype=torch.bool)[None]
        weight=weight*valid
        geom=torch.stack([a/self.D,cross/self.D,dz.expand_as(a)/self.D,weight],-1)
        glob=torch.stack([ws.mean(1)/20,psi.sin(),psi.cos()],-1)
        return weight,geom,glob

    @staticmethod
    def _gather(h,idx):
        batch=torch.arange(h.shape[0],device=h.device)[:,None,None]
        return h[batch,idx]

    def forward(self,x,nbr_idx=None,edge_attr=None,edge_mask=None,role=None):
        B,L,N,F=x.shape
        if L!=self.history:
            raise ValueError("STV baseline history differs from the original configuration")
        weight,geom,glob=self._physics(x[:,-1])
        weight,idx=weight.topk(min(self.top_k,N),dim=2)
        K=idx.shape[2]
        geom=geom.gather(2,idx[...,None].expand(-1,-1,-1,4))
        h=self.node_in(x.permute(0,2,1,3).reshape(B,N,L*F))
        g,e=self.global_in(glob),self.edge_in(geom)
        for eu,nu,gu,norm in zip(self.edge_updates,self.node_updates,self.global_updates,self.norms):
            src=self._gather(h,idx)
            dst=h[:,:,None,:].expand(-1,-1,K,-1)
            gg=g[:,None,None,:].expand(-1,N,K,-1)
            e=eu(torch.cat([e,src,dst,gg],-1))*weight[...,None]
            h=norm(h+nu(torch.cat([e.sum(2),h,g[:,None,:].expand(-1,N,-1)],-1)))
            g=gu(torch.cat([e.sum((1,2)),h.sum(1),g],-1))
        return torch.sigmoid(self.head(h))*1.5


def make_model(cfg: Config, mode: str, in_dim: int, n_turbines: int = None) -> nn.Module:
    """
    模型/消融说明：
    - full:             TCN + 三尾流物理图 + 软异构门控 + 自适应图残差分支
    - full_no_adapt:    TCN + 三尾流物理图 + 软异构门控，去掉自适应图
    - no_hetero_adapt:  TCN + 三尾流物理图 + 自适应图，去掉软异构门控
    - adaptive_only:    TCN + 自适应图，去掉三尾流物理图和软异构门控
    - no_hetero:        TCN + 三尾流物理图，去掉软异构门控和自适应图
    - no_graph:         只保留 TCN，不使用任何图结构
    """
    out_steps = int(max(1, cfg.pred_horizon))
    n_nodes = int(n_turbines) if n_turbines is not None else int(max(1, cfg.max_turbines))

    if mode == "stv_pgnn_adapted":
        return STVPGNNAdaptedBaseline(cfg, in_dim, n_nodes)

    if mode == "no_graph":
        return LocalTCNMLP(
            in_dim,
            cfg.hidden_dim,
            cfg.num_layers,
            cfg.dropout,
            tcn_layers=cfg.tcn_layers,
            tcn_kernel_size=cfg.tcn_kernel_size,
            out_steps=out_steps,
        )

    # 本文方法及其关键消融
    if mode == "full":
        return LightweightWakeGAT(
            in_dim,
            cfg.hidden_dim,
            cfg.gat_heads,
            cfg.dropout,
            use_soft_hetero=True,
            hetero_gate_alpha=float(cfg.hetero_gate_alpha),
            tcn_layers=cfg.tcn_layers,
            tcn_kernel_size=cfg.tcn_kernel_size,
            out_steps=out_steps,
            use_adaptive_graph=True,
            n_nodes=n_nodes,
            adaptive_graph_emb_dim=int(cfg.adaptive_graph_emb_dim),
            adaptive_graph_alpha=float(cfg.adaptive_graph_alpha),
            adaptive_graph_top_k=int(cfg.adaptive_graph_top_k),
            use_wake_delay=bool(int(cfg.use_wake_delay)),
            wake_delay_max_steps=int(cfg.wake_delay_max_steps),
            use_learnable_fusion=bool(int(cfg.use_learnable_fusion)),
        )
    if mode == "full_no_adapt":
        return LightweightWakeGAT(
            in_dim,
            cfg.hidden_dim,
            cfg.gat_heads,
            cfg.dropout,
            use_soft_hetero=True,
            hetero_gate_alpha=float(cfg.hetero_gate_alpha),
            tcn_layers=cfg.tcn_layers,
            tcn_kernel_size=cfg.tcn_kernel_size,
            out_steps=out_steps,
            use_adaptive_graph=False,
            use_wake_delay=bool(int(cfg.use_wake_delay)),
            wake_delay_max_steps=int(cfg.wake_delay_max_steps),
            use_learnable_fusion=bool(int(cfg.use_learnable_fusion)),
        )
    if mode == "no_hetero_adapt":
        return LightweightWakeGAT(
            in_dim,
            cfg.hidden_dim,
            cfg.gat_heads,
            cfg.dropout,
            use_soft_hetero=False,
            hetero_gate_alpha=float(cfg.hetero_gate_alpha),
            tcn_layers=cfg.tcn_layers,
            tcn_kernel_size=cfg.tcn_kernel_size,
            out_steps=out_steps,
            use_adaptive_graph=True,
            n_nodes=n_nodes,
            adaptive_graph_emb_dim=int(cfg.adaptive_graph_emb_dim),
            adaptive_graph_alpha=float(cfg.adaptive_graph_alpha),
            adaptive_graph_top_k=int(cfg.adaptive_graph_top_k),
            use_wake_delay=bool(int(cfg.use_wake_delay)),
            wake_delay_max_steps=int(cfg.wake_delay_max_steps),
            use_learnable_fusion=bool(int(cfg.use_learnable_fusion)),
        )
    if mode == "adaptive_only":
        # build_dataset 中 adaptive_only 不构建三尾流物理图；因此物理 WakeGAT 消息为 0，
        # 这里只保留 TCN 编码、自适应图残差分支和 FFN/head。
        return LightweightWakeGAT(
            in_dim,
            cfg.hidden_dim,
            cfg.gat_heads,
            cfg.dropout,
            use_soft_hetero=False,
            hetero_gate_alpha=float(cfg.hetero_gate_alpha),
            tcn_layers=cfg.tcn_layers,
            tcn_kernel_size=cfg.tcn_kernel_size,
            out_steps=out_steps,
            use_adaptive_graph=True,
            n_nodes=n_nodes,
            adaptive_graph_emb_dim=int(cfg.adaptive_graph_emb_dim),
            adaptive_graph_alpha=float(cfg.adaptive_graph_alpha),
            adaptive_graph_top_k=int(cfg.adaptive_graph_top_k),
            use_wake_delay=bool(int(cfg.use_wake_delay)),
            wake_delay_max_steps=int(cfg.wake_delay_max_steps),
            use_learnable_fusion=bool(int(cfg.use_learnable_fusion)),
        )
    if mode == "no_hetero":
        return LightweightWakeGAT(
            in_dim,
            cfg.hidden_dim,
            cfg.gat_heads,
            cfg.dropout,
            use_soft_hetero=False,
            hetero_gate_alpha=float(cfg.hetero_gate_alpha),
            tcn_layers=cfg.tcn_layers,
            tcn_kernel_size=cfg.tcn_kernel_size,
            out_steps=out_steps,
            use_adaptive_graph=False,
            use_wake_delay=bool(int(cfg.use_wake_delay)),
            wake_delay_max_steps=int(cfg.wake_delay_max_steps),
            use_learnable_fusion=bool(int(cfg.use_learnable_fusion)),
        )

    # 常用时序/时空对比模型
    if mode == "lstm":
        return SharedLSTMBaseline(in_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, out_steps=out_steps)
    if mode == "cnn_lstm":
        return CNNLSTMBaseline(in_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, out_steps=out_steps)
    if mode == "cnn_bilstm":
        return CNNBiLSTMBaseline(in_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, out_steps=out_steps)
    if mode == "transformer":
        return TemporalTransformerBaseline(in_dim, cfg.hidden_dim, cfg.gat_heads, cfg.num_layers, cfg.dropout, out_steps=out_steps)
    if mode == "dbfp_net":
        return LightweightDBFPNetBaseline(in_dim, cfg.hidden_dim, cfg.dropout, out_steps=out_steps)
    if mode == "stgcn":
        return STGCNBaseline(in_dim, cfg.hidden_dim, cfg.dropout, cfg.tcn_layers, cfg.tcn_kernel_size, out_steps=out_steps)
    if mode == "agcrn":
        return AGCRNBaseline(in_dim, n_nodes, cfg.hidden_dim, cfg.dropout, out_steps=out_steps)
    if mode == "mtgnn":
        return MTGNNBaseline(in_dim, n_nodes, cfg.hidden_dim, cfg.dropout, out_steps=out_steps)

    raise ValueError(f"未知 mode: {mode}")

def evaluate(model: nn.Module, data: Dict[str, np.ndarray], cfg: Config, n_turbines: int) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    preds = []
    ys = []
    bs_time = max(1, int(cfg.batch_size) // max(n_turbines, 1))
    with torch.no_grad():
        T = data["X_time"].shape[0]
        for i in range(0, T, bs_time):
            xb = torch.from_numpy(data["X_time"][i:i + bs_time]).float().to(cfg.device)
            yb = torch.from_numpy(data["Y_time"][i:i + bs_time]).float().to(cfg.device)
            nbr = torch.from_numpy(data["nbr_idx"][i:i + bs_time]).to(cfg.device)
            edge = torch.from_numpy(data["edge_attr"][i:i + bs_time]).to(cfg.device)
            mask = torch.from_numpy(data["edge_mask"][i:i + bs_time]).to(cfg.device)
            role = torch.from_numpy(data["role"][i:i + bs_time]).float().to(cfg.device)
            p = model(xb, nbr, edge, mask, role).detach().cpu().numpy()
            preds.append(p)
            ys.append(yb.detach().cpu().numpy())

    pred_norm = np.concatenate(preds, axis=0)  # [S,N,H]
    true_norm = np.concatenate(ys, axis=0)     # [S,N,H]
    pred_kw = np.clip(pred_norm, 0.0, 1.5) * float(cfg.rated_power_kw)
    true_kw = np.clip(true_norm, 0.0, 1.5) * float(cfg.rated_power_kw)
    H = int(pred_kw.shape[-1])

    out: Dict[str, float] = {}
    # 每个预测步长分别计算指标：h1=t+1, h2=t+2, ..., hH=t+H。
    for hh in range(H):
        step = hh + 1
        node_metrics = compute_metrics(
            true_kw[:, :, hh].reshape(-1),
            pred_kw[:, :, hh].reshape(-1),
            floor_kw=100.0,
            capacity_kw=float(cfg.rated_power_kw),
        )
        farm_metrics = compute_metrics(
            true_kw[:, :, hh].sum(axis=1),
            pred_kw[:, :, hh].sum(axis=1),
            floor_kw=max(1500.0, 0.08 * n_turbines * cfg.rated_power_kw),
            capacity_kw=float(n_turbines) * float(cfg.rated_power_kw),
        )
        out.update({f"node_h{step}_{k}": v for k, v in node_metrics.items()})
        out.update({f"farm_h{step}_{k}": v for k, v in farm_metrics.items()})

    # 1..H 步整体平均指标：把所有预测步长展开后统一计算。
    node_avg_metrics = compute_metrics(
        true_kw.reshape(-1),
        pred_kw.reshape(-1),
        floor_kw=100.0,
        capacity_kw=float(cfg.rated_power_kw),
    )
    farm_true_all = true_kw.sum(axis=1)  # [S,H]
    farm_pred_all = pred_kw.sum(axis=1)  # [S,H]
    farm_avg_metrics = compute_metrics(
        farm_true_all.reshape(-1),
        farm_pred_all.reshape(-1),
        floor_kw=max(1500.0, 0.08 * n_turbines * cfg.rated_power_kw),
        capacity_kw=float(n_turbines) * float(cfg.rated_power_kw),
    )
    out.update({f"node_avg_{k}": v for k, v in node_avg_metrics.items()})
    out.update({f"farm_avg_{k}": v for k, v in farm_avg_metrics.items()})
    return out, pred_kw


def train_one_experiment(cfg: Config, mode: str) -> Dict[str, float]:
    set_seed(cfg.seed)
    train, val, test, turbines, _ = build_dataset(cfg, mode)
    n_turbines = len(turbines)
    in_dim = int(train["X_time"].shape[-1])

    model = make_model(cfg, mode, in_dim, n_turbines=n_turbines).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.MSELoss()

    time_batch_size = max(1, int(cfg.batch_size) // max(n_turbines, 1))
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["X_time"]).float(),
            torch.from_numpy(train["Y_time"]).float(),
            torch.from_numpy(train["nbr_idx"]),
            torch.from_numpy(train["edge_attr"]),
            torch.from_numpy(train["edge_mask"]),
            torch.from_numpy(train["role"]).float(),
        ),
        batch_size=time_batch_size, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=(cfg.device == "cuda"),
    )
    model_name_map = {
        "stv_pgnn_adapted": "STV-PGNN-adapted",
        "full": "TCN-DelayPhysSoftHeteroWakeGAT-AdaptiveFusion",
        "full_no_adapt": "TCN-PhysSoftHeteroWakeGAT",
        "no_hetero_adapt": "TCN-PhysWakeGAT-Adaptive",
        "adaptive_only": "TCN-AdaptiveGraph",
        "no_hetero": "TCN-PhysWakeGAT",
        "no_graph": "TCN",
        "lstm": "LSTM",
        "cnn_lstm": "CNN-LSTM",
        "transformer": "Transformer",
        "dbfp_net": "DBFP-Net",
        "cnn_bilstm": "CNN-BiLSTM",
        "stgcn": "STGCN",
        "agcrn": "AGCRN",
        "mtgnn": "MTGNN",
    }
    model_name = model_name_map.get(mode, mode)
    print(
        f"[{mode}] model={model_name}, time_batch_size={time_batch_size}, "
        f"window={int(cfg.temporal_window)}, tcn_layers={int(cfg.tcn_layers)}, "
        f"heads={int(cfg.gat_heads)}, top_k={int(cfg.gat_top_k)}, "
        f"hetero_gate_alpha={float(cfg.hetero_gate_alpha):.3f}, "
        f"adaptive_alpha={float(cfg.adaptive_graph_alpha):.3f}, "
        f"adaptive_top_k={int(cfg.adaptive_graph_top_k)}, "
        f"delay={int(cfg.use_wake_delay)}, delay_max={int(cfg.wake_delay_max_steps)}, "
        f"fusion={int(cfg.use_learnable_fusion)}, "
        f"farm_loss_weight={float(cfg.farm_loss_weight):.4f}"
    )

    best_val = float("inf")
    best_state = None
    bad = 0
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses, node_losses, farm_losses = [], [], []
        for xb, yb, nbr, edge, mask, role in train_loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)
            nbr = nbr.to(cfg.device, non_blocking=True)
            edge = edge.to(cfg.device, non_blocking=True)
            mask = mask.to(cfg.device, non_blocking=True)
            role = role.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(xb, nbr, edge, mask, role)
            node_loss = loss_fn(pred, yb)
            pred_farm = pred.mean(dim=1)
            true_farm = yb.mean(dim=1)
            farm_loss = loss_fn(pred_farm, true_farm)
            loss = node_loss + float(cfg.farm_loss_weight) * farm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            node_losses.append(float(node_loss.detach().cpu()))
            farm_losses.append(float(farm_loss.detach().cpu()))

        val_metrics, _ = evaluate(model, val, cfg, n_turbines)
        val_rmse = float(val_metrics["node_avg_rmse"])
        history.append({
            "epoch": ep,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            "train_node_loss": float(np.mean(node_losses)) if node_losses else np.nan,
            "train_farm_loss": float(np.mean(farm_losses)) if farm_losses else np.nan,
            "val_node_avg_rmse": val_rmse,
            "val_farm_avg_rmse": float(val_metrics["farm_avg_rmse"]),
        })
        print(
            f"[{mode}] epoch {ep:03d} | "
            f"loss={history[-1]['train_loss']:.6f} | node_loss={history[-1]['train_node_loss']:.6f} | farm_loss={history[-1]['train_farm_loss']:.6f} | "
            f"val_node_avg_RMSE={val_rmse:.3f} kW | val_farm_avg_RMSE={float(val_metrics['farm_avg_rmse']):.3f} kW"
        )

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(cfg.patience):
                print(f"[{mode}] early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, _ = evaluate(model, val, cfg, n_turbines)
    test_metrics, pred_matrix_kw = evaluate(model, test, cfg, n_turbines)

    os.makedirs(cfg.out_dir, exist_ok=True)
    hist_path = os.path.join(cfg.out_dir, f"history_{mode}.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False, encoding="utf-8-sig")

    true_matrix_kw = test["Y_matrix"] * float(cfg.rated_power_kw)  # [S,N,H]
    rows = []
    H = int(true_matrix_kw.shape[-1])
    for ti in range(true_matrix_kw.shape[0]):
        for hh in range(H):
            tstamp = pd.Timestamp(test["times"][ti, hh])
            for j, turb in enumerate(turbines):
                rows.append({
                    "target_time": tstamp,
                    "horizon": hh + 1,
                    "turbine": turb,
                    "true_kw": float(true_matrix_kw[ti, j, hh]),
                    "pred_kw": float(pred_matrix_kw[ti, j, hh]),
                })
    pred_path = os.path.join(cfg.out_dir, f"test_predictions_{mode}.csv")
    pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8-sig")

    result = {"case": mode, "model": model_name, "feature_dim": in_dim, "best_val_node_avg_rmse": best_val}
    result.update({f"val_{k}": v for k, v in val_metrics.items()})
    result.update({f"test_{k}": v for k, v in test_metrics.items()})
    H = int(max(1, cfg.pred_horizon))
    for hh in range(1, H + 1):
        print(
            f"[{mode}] Test h{hh} node RMSE={test_metrics[f'node_h{hh}_rmse']:.3f}, "
            f"MAE={test_metrics[f'node_h{hh}_mae']:.3f}, "
            f"nRMSE={test_metrics[f'node_h{hh}_nrmse']:.3f}%, "
            f"nMAE={test_metrics[f'node_h{hh}_nmae']:.3f}%, "
            f"WAPE={test_metrics[f'node_h{hh}_wape']:.3f}%, "
            f"R2={test_metrics[f'node_h{hh}_r2']:.4f}"
        )
        print(
            f"[{mode}] Test h{hh} farm RMSE={test_metrics[f'farm_h{hh}_rmse']:.3f}, "
            f"MAE={test_metrics[f'farm_h{hh}_mae']:.3f}, "
            f"nRMSE={test_metrics[f'farm_h{hh}_nrmse']:.3f}%, "
            f"nMAE={test_metrics[f'farm_h{hh}_nmae']:.3f}%, "
            f"WAPE={test_metrics[f'farm_h{hh}_wape']:.3f}%, "
            f"R2={test_metrics[f'farm_h{hh}_r2']:.4f}"
        )
    print(
        f"[{mode}] Test AVG node RMSE={test_metrics['node_avg_rmse']:.3f}, MAE={test_metrics['node_avg_mae']:.3f}, "
        f"nRMSE={test_metrics['node_avg_nrmse']:.3f}%, nMAE={test_metrics['node_avg_nmae']:.3f}%, "
        f"WAPE={test_metrics['node_avg_wape']:.3f}%, R2={test_metrics['node_avg_r2']:.4f}"
    )
    print(
        f"[{mode}] Test AVG farm RMSE={test_metrics['farm_avg_rmse']:.3f}, MAE={test_metrics['farm_avg_mae']:.3f}, "
        f"nRMSE={test_metrics['farm_avg_nrmse']:.3f}%, nMAE={test_metrics['farm_avg_nmae']:.3f}%, "
        f"WAPE={test_metrics['farm_avg_wape']:.3f}%, R2={test_metrics['farm_avg_r2']:.4f}"
    )
    return result


# =========================
# 7. 主函数
# =========================

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Soft heterogeneous dynamic wake graph + three wake formulas + lightweight Wake-GAT for SDWPF power prediction")
    p.add_argument("--data_csv", type=str, default=Config.data_csv)
    p.add_argument("--location_csv", type=str, default=Config.location_csv)
    p.add_argument("--out_dir", type=str, default=Config.out_dir)
    p.add_argument("--max_turbines", type=int, default=Config.max_turbines)
    p.add_argument("--rated_power_kw", type=float, default=Config.rated_power_kw)
    p.add_argument("--pred_horizon", type=int, default=Config.pred_horizon)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--hidden_dim", type=int, default=Config.hidden_dim)
    p.add_argument("--num_layers", type=int, default=Config.num_layers)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--gat_heads", type=int, default=Config.gat_heads)
    p.add_argument("--gat_top_k", type=int, default=Config.gat_top_k)
    p.add_argument("--temporal_window", type=int, default=Config.temporal_window,
                   help="Length of historical window for TCN temporal encoding.")
    p.add_argument("--tcn_layers", type=int, default=Config.tcn_layers)
    p.add_argument("--tcn_kernel_size", type=int, default=Config.tcn_kernel_size)
    p.add_argument("--hetero_gate_alpha", type=float, default=Config.hetero_gate_alpha,
                   help="Strength of soft heterogeneous role gate in Wake-GAT attention. Affects full and full_no_adapt.")
    p.add_argument("--adaptive_graph_alpha", type=float, default=Config.adaptive_graph_alpha,
                   help="Residual weight of the adaptive graph branch. Affects full, no_hetero_adapt and adaptive_only.")
    p.add_argument("--adaptive_graph_emb_dim", type=int, default=Config.adaptive_graph_emb_dim,
                   help="Node embedding dimension for the adaptive graph branch. Affects full, no_hetero_adapt and adaptive_only.")
    p.add_argument("--adaptive_graph_top_k", type=int, default=Config.adaptive_graph_top_k,
                   help="Top-k neighbors kept in the adaptive graph. 0 means dense adaptive graph. Affects full, no_hetero_adapt and adaptive_only.")
    p.add_argument("--use_wake_delay", type=int, default=Config.use_wake_delay,
                   help="Whether to use delay-aware physical wake message propagation. 1=yes, 0=no.")
    p.add_argument("--sample_interval_minutes", type=float, default=Config.sample_interval_minutes,
                   help="Sampling interval in minutes for wake propagation delay estimation.")
    p.add_argument("--wake_delay_speed_floor", type=float, default=Config.wake_delay_speed_floor,
                   help="Minimum wind speed used in delay estimation to avoid unrealistically large delays.")
    p.add_argument("--wake_delay_max_steps", type=int, default=Config.wake_delay_max_steps,
                   help="Maximum wake propagation delay in historical sampling steps.")
    p.add_argument("--use_learnable_fusion", type=int, default=Config.use_learnable_fusion,
                   help="Whether to use node-wise learnable fusion gate between physics and adaptive branches. 1=yes, 0=fixed alpha.")
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--farm_loss_weight", type=float, default=Config.farm_loss_weight,
                   help="Farm-level auxiliary loss weight: loss = node_loss + lambda * farm_loss")
    p.add_argument("--patience", type=int, default=Config.patience)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--experiments", type=str, default=Config.experiments,
                   help="逗号分隔：full,full_no_adapt,no_hetero_adapt,adaptive_only,no_hetero,no_graph,lstm,cnn_lstm,transformer,dbfp_net,cnn_bilstm,stgcn,agcrn,mtgnn,stv_pgnn_adapted。")
    args = p.parse_args()
    return Config(**vars(args))


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("TCN + 动态尾流启发图 + 延迟注意力 + 自适应图")
    print(f"device={cfg.device}, out_dir={cfg.out_dir}")
    print(f"实验：{cfg.experiments}；预测步长：1..{cfg.pred_horizon}")
    print("=" * 80)

    allowed = {"full", "full_no_adapt", "no_hetero_adapt", "adaptive_only", "no_hetero", "no_graph", "lstm", "cnn_lstm", "transformer", "dbfp_net", "cnn_bilstm", "stgcn", "agcrn", "mtgnn", "stv_pgnn_adapted"}
    exps = [e.strip() for e in str(cfg.experiments).split(",") if e.strip()]
    for e in exps:
        if e not in allowed:
            raise ValueError(f"未知实验 {e}，只能从 {sorted(allowed)} 中选择。")

    results = []
    for mode in exps:
        print("\n" + "#" * 80)
        print(f"开始实验: {mode}")
        print("#" * 80)
        res = train_one_experiment(cfg, mode)
        results.append(res)

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(cfg.out_dir, "summary_results.csv")
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 论文主表用长表：每个模型、每个预测步长一行，只显示 RMSE/MAE/nRMSE/nMAE/WAPE/R2。
    H = int(max(1, cfg.pred_horizon))
    paper_rows = []
    for res in results:
        for hh in range(1, H + 1):
            paper_rows.append({
                "case": res["case"],
                "model": res["model"],
                "horizon": f"t+{hh}",
                "node_RMSE": res[f"test_node_h{hh}_rmse"],
                "node_MAE": res[f"test_node_h{hh}_mae"],
                "node_nRMSE(%)": res[f"test_node_h{hh}_nrmse"],
                "node_nMAE(%)": res[f"test_node_h{hh}_nmae"],
                "node_WAPE(%)": res[f"test_node_h{hh}_wape"],
                "node_R2": res[f"test_node_h{hh}_r2"],
                "farm_RMSE": res[f"test_farm_h{hh}_rmse"],
                "farm_MAE": res[f"test_farm_h{hh}_mae"],
                "farm_nRMSE(%)": res[f"test_farm_h{hh}_nrmse"],
                "farm_nMAE(%)": res[f"test_farm_h{hh}_nmae"],
                "farm_WAPE(%)": res[f"test_farm_h{hh}_wape"],
                "farm_R2": res[f"test_farm_h{hh}_r2"],
            })
        paper_rows.append({
            "case": res["case"],
            "model": res["model"],
            "horizon": "avg",
            "node_RMSE": res["test_node_avg_rmse"],
            "node_MAE": res["test_node_avg_mae"],
            "node_nRMSE(%)": res["test_node_avg_nrmse"],
            "node_nMAE(%)": res["test_node_avg_nmae"],
            "node_WAPE(%)": res["test_node_avg_wape"],
            "node_R2": res["test_node_avg_r2"],
            "farm_RMSE": res["test_farm_avg_rmse"],
            "farm_MAE": res["test_farm_avg_mae"],
            "farm_nRMSE(%)": res["test_farm_avg_nrmse"],
            "farm_nMAE(%)": res["test_farm_avg_nmae"],
            "farm_WAPE(%)": res["test_farm_avg_wape"],
            "farm_R2": res["test_farm_avg_r2"],
        })

    df_paper = pd.DataFrame(paper_rows)
    paper_csv = os.path.join(cfg.out_dir, "summary_results_by_horizon.csv")
    df_paper.to_csv(paper_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("全部实验完成，按预测步长汇总结果：")
    print(df_paper.to_string(index=False))
    print(f"完整宽表已保存: {out_csv}")
    print(f"论文主表长表已保存: {paper_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
