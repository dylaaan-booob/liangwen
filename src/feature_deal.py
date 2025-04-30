import pandas as pd
import numpy as np

def safe_divide(numerator, denominator, default=0.0):
    result = np.full_like(numerator, default, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result

def time_feature(df, name="time"):
    t = pd.to_datetime(df[name], format='%H:%M:%S')
    hours = t.dt.hour
    minutes = t.dt.minute
    seconds = t.dt.second
    df['time_frac'] = (hours * 3600 + minutes * 60 + seconds) / 86400.0

    # 3) 周期性编码
    df['sin_time'] = np.sin(2 * np.pi * df['time_frac'])
    df['cos_time'] = np.cos(2 * np.pi * df['time_frac'])
    return df, ["sin_time", "cos_time", "time_frac"]

def stock_feature(df):
    bid_cols = ['n_bid1', 'n_bid2', 'n_bid3', 'n_bid4', 'n_bid5']
    ask_cols = ['n_ask1', 'n_ask2', 'n_ask3', 'n_ask4', 'n_ask5']
    bsize_cols = ['n_bsize1', 'n_bsize2', 'n_bsize3', 'n_bsize4', 'n_bsize5']
    asize_cols = ['n_asize1', 'n_asize2', 'n_asize3', 'n_asize4', 'n_asize5']

    # 矢量化操作，避免逐列操作
    df[bid_cols] += 1
    df[ask_cols] += 1

    # 使用 NumPy 进行矢量化计算
    bid_values = df[bid_cols].values
    ask_values = df[ask_cols].values
    bsize_values = df[bsize_cols].values
    asize_values = df[asize_cols].values
    high_values = df["n_high"].values
    low_values = df["n_low"].values
    close_values = df["n_close"].values
    open_values = df["n_open"].values

    mid_values = df["n_midprice"].values
    # 计算 spread 和 mid_price
    # spread越小，市场流动性越高
    # spread越大，交易成本越大
    spread = ask_values - bid_values

    # mid price是市场公允价格估计，更加接近理论价格
    mid_price = (ask_values + bid_values) / 2

    # 计算 weighted_ab
    # 这个是计算买卖差值，衡量买卖方向上的倾斜
    weighted_ab = (ask_values * bsize_values - bid_values * asize_values) / (bsize_values + asize_values)

    # 计算 vol1_rel_diff 和 volall_rel_diff
    # 这是一档买卖量不对称性，通过买一量 - 卖一量 / 买一量 + 卖一量，值趋近1代表买入压力大，会上涨，
    vol1_rel_diff = (bsize_values[:, 0] - asize_values[:, 0]) / (bsize_values[:, 0] + asize_values[:, 0])
    # 这是总买辆对称性，衡量整个五档的订单流失衡，1就是买入压力，-1就是卖出压力
    volall_rel_diff = (bsize_values.sum(axis=1) - asize_values.sum(axis=1)) / (
                bsize_values.sum(axis=1) + asize_values.sum(axis=1))
    df['spread1'], df['spread2'], df['spread3'] = spread[:, 0], spread[:, 1], spread[:, 2]
    df['mid_price1'], df['mid_price2'], df['mid_price3'] = mid_price[:, 0], mid_price[:, 1], mid_price[:, 2]
    df['weighted_ab1'], df['weighted_ab2'], df['weighted_ab3'] = weighted_ab[:, 0], weighted_ab[:, 1], weighted_ab[:, 2]
    df['vol1_rel_diff'] = vol1_rel_diff
    df['volall_rel_diff'] = volall_rel_diff
    df["depth_bid"] = bsize_values.sum(axis=1)
    df["depth_ask"] = asize_values.sum(axis=1)
    df["amplitude"] = safe_divide(high_values - low_values, open_values)

    # 当前涨跌 now_up_down
    df["now_up_down"] = safe_divide(close_values - open_values, open_values, 0.0)

    # 价格位置 now_location
    df["now_location"] = safe_divide(close_values - low_values, high_values - low_values, 0.5)

    # 偏离中间价比例 bias_midprice
    df["bias_midprice"] = safe_divide(close_values - mid_values, mid_values, 0.0)

    # 威廉指标 william_R
    df["william_R"] = safe_divide(high_values - close_values, high_values - low_values, -50.0)
    # 对 amount_delta 应用 np.log1p
    df['amount'] = np.log1p(df['amount_delta'].values)
    # 返回所需的特征列
    feature_col_names = [
        'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
        'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1',
        'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4',
        'n_ask5', 'n_asize5', 'spread1', 'mid_price1', 'spread2', 'mid_price2',
        'spread3', 'mid_price3', 'weighted_ab1', 'weighted_ab2', 'weighted_ab3',
        'amount', 'vol1_rel_diff', 'volall_rel_diff', "depth_ask", "depth_bid",
        "amplitude", "now_up_down", "now_location", "bias_midprice", "william_R"]
    return df, feature_col_names

def main_feature(df, only_columns=False):
    """返回df一定要保留sym列，之后要使用对应sym的模型进行预测"""
    total = []
    if only_columns:
        df = df.iloc[:1].copy()
    for func in [time_feature, stock_feature]:
        df, t = func(df)
        total.extend(t)
    if only_columns:
        return total
    return df[total + ["sym"]]



