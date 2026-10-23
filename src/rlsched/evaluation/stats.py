"""Thống kê cho so sánh phương pháp trên CÙNG các seed test (thiết kế ghép cặp / paired).

- mean ± std, khoảng tin cậy 95% (bootstrap percentile) qua các episode test
- kiểm định Wilcoxon signed-rank (chính) và paired t-test (tham khảo) cho từng cặp phương pháp
- hiệu chỉnh Holm–Bonferroni khi so sánh nhiều cặp
- cải thiện tương đối (%) = (baseline - method) / baseline * 100 (metric càng thấp càng tốt)
Với phương pháp RL có nhiều seed huấn luyện: lấy trung bình theo seed huấn luyện cho từng episode test
trước khi kiểm định (đơn vị độc lập là chuỗi job test), đồng thời báo cáo độ lệch giữa các seed huấn luyện.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def bootstrap_ci(x, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def summarize(episodes: pd.DataFrame, metric: str = "avg_jct_s", by: str = "method") -> pd.DataFrame:
    rows = []
    for name, g in episodes.groupby(by, sort=False):
        x = g[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(x)
        rows.append({by: name, "n_episodes": int(np.isfinite(x).sum()), "mean": float(np.nanmean(x)),
                     "std": float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan,
                     "ci95_low": lo, "ci95_high": hi, "median": float(np.nanmedian(x))})
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_compare(episodes: pd.DataFrame, method: str, baselines: list[str], metric: str = "avg_jct_s",
                   key: str = "seed", by: str = "method") -> pd.DataFrame:
    """So sánh `method` với từng baseline trên cùng các seed test."""
    wide = episodes.pivot_table(index=key, columns=by, values=metric, aggfunc="mean")
    rows = []
    for b in baselines:
        if b == method or b not in wide or method not in wide:
            continue
        pair = wide[[method, b]].dropna()
        d = pair[b].to_numpy() - pair[method].to_numpy()  # >0: method tốt hơn (metric thấp hơn)
        try:
            w = sps.wilcoxon(pair[method], pair[b], zero_method="wilcox", alternative="two-sided")
            p_w = float(w.pvalue)
        except ValueError:
            p_w = np.nan
        t = sps.ttest_rel(pair[method], pair[b])
        rel = 100.0 * d / pair[b].to_numpy()
        lo, hi = bootstrap_ci(rel)
        rows.append({"method": method, "baseline": b, "n_pairs": len(pair),
                     "mean_diff": float(d.mean()), "rel_improvement_pct": float(rel.mean()),
                     "rel_ci95_low": lo, "rel_ci95_high": hi,
                     "win_rate": float((d > 0).mean()), "p_wilcoxon": p_w, "p_ttest": float(t.pvalue)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].fillna(1.0).tolist())
    return out


def aggregate_train_seeds(episodes: pd.DataFrame, method_col: str = "method", group_col: str = "algo",
                          metric: str = "avg_jct_s") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Với các run RL nhãn kiểu 'Decima/s0', 'Decima/s1' (cột algo='Decima', train_seed=0,1,..):
    trả về (1) bảng episode đã lấy trung bình theo seed huấn luyện, (2) mean ± std giữa các seed huấn luyện."""
    per_seed = episodes.groupby([group_col, "train_seed"])[metric].mean().reset_index()
    spread = per_seed.groupby(group_col)[metric].agg(["mean", "std", "count"]).reset_index()
    spread = spread.rename(columns={"count": "n_train_seeds"})
    avg = episodes.groupby([group_col, "seed"])[metric].mean().reset_index().rename(columns={group_col: method_col})
    return avg, spread
