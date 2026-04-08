"""在线累计统计（用于 WebUI 推送）。"""

from __future__ import annotations

from dataclasses import dataclass


class P2Quantile:
    """P² (P-square) 在线分位数估计器（Jain & Chlamtac, 1985）。

    - 常量内存、单次更新 O(1)
    - 适用于长流式数据的近似分位数
    - 本实现用于 p99（q=0.99）一类指标
    """

    def __init__(self, q: float) -> None:
        if not (0.0 < q < 1.0):
            raise ValueError("q must be in (0,1)")
        self.q = float(q)
        self._init: list[float] = []
        self._n = [0, 0, 0, 0, 0]  # marker positions
        self._np = [0.0, 0.0, 0.0, 0.0, 0.0]  # desired positions
        self._dn = [0.0, 0.0, 0.0, 0.0, 0.0]  # increments
        self._qv = [0.0, 0.0, 0.0, 0.0, 0.0]  # marker heights (values)

    def _bootstrap(self) -> None:
        self._init.sort()
        self._qv = self._init[:5]
        # positions are 1-indexed in paper; we keep 1-indexed too
        self._n = [1, 2, 3, 4, 5]
        q = self.q
        self._np = [1.0, 1.0 + 2.0 * q, 1.0 + 4.0 * q, 3.0 + 2.0 * q, 5.0]
        self._dn = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]

    def update(self, x: float) -> None:
        v = float(x)
        if len(self._init) < 5:
            self._init.append(v)
            if len(self._init) == 5:
                self._bootstrap()
            return

        # find k: the cell in which x falls
        if v < self._qv[0]:
            self._qv[0] = v
            k = 0
        elif v < self._qv[1]:
            k = 0
        elif v < self._qv[2]:
            k = 1
        elif v < self._qv[3]:
            k = 2
        elif v < self._qv[4]:
            k = 3
        else:
            self._qv[4] = v
            k = 3

        # increment positions of markers k+1..5
        for i in range(k + 1, 5):
            self._n[i] += 1
        # update desired positions
        for i in range(5):
            self._np[i] += self._dn[i]

        # adjust heights of markers 2..4 if necessary
        for i in (1, 2, 3):
            d = self._np[i] - float(self._n[i])
            if (d >= 1.0 and self._n[i + 1] - self._n[i] > 1) or (
                d <= -1.0 and self._n[i - 1] - self._n[i] < -1
            ):
                di = 1 if d > 0 else -1
                # parabolic prediction
                qi = self._qv[i]
                q_im1 = self._qv[i - 1]
                q_ip1 = self._qv[i + 1]
                n_i = float(self._n[i])
                n_im1 = float(self._n[i - 1])
                n_ip1 = float(self._n[i + 1])

                num = di * (
                    (n_i - n_im1 + di) * (q_ip1 - qi) / (n_ip1 - n_i)
                    + (n_ip1 - n_i - di) * (qi - q_im1) / (n_i - n_im1)
                )
                den = (n_ip1 - n_im1)
                q_new = qi + (num / den)

                # if parabolic goes out of bounds, use linear
                if q_im1 < q_new < q_ip1:
                    self._qv[i] = q_new
                else:
                    self._qv[i] = qi + di * (self._qv[i + di] - qi) / (
                        float(self._n[i + di]) - n_i
                    )
                self._n[i] += di

    def value(self) -> float | None:
        if len(self._init) == 0:
            return None
        if len(self._init) < 5:
            # exact quantile on small set
            xs = sorted(self._init)
            if len(xs) == 1:
                return xs[0]
            # nearest-rank
            k = int(round(self.q * (len(xs) - 1)))
            k = max(0, min(len(xs) - 1, k))
            return xs[k]
        return float(self._qv[2])  # marker 3 tracks desired quantile


@dataclass
class RunningErrorStats:
    """累计误差统计：对所有 step 的所有动作维度展开后的流进行统计。"""

    rel_sum: float = 0.0
    rel_count: int = 0
    p99_abs: P2Quantile | None = None

    def __post_init__(self) -> None:
        if self.p99_abs is None:
            self.p99_abs = P2Quantile(0.99)

    def update_abs_and_rel(self, *, abs_err_values, rel_err_values) -> None:
        # abs_err_values / rel_err_values: iterable[float]
        for v in abs_err_values:
            self.p99_abs.update(float(v))
        for v in rel_err_values:
            self.rel_sum += float(v)
            self.rel_count += 1

    def mean_rel(self) -> float | None:
        if self.rel_count <= 0:
            return None
        return float(self.rel_sum / float(self.rel_count))

    def p99_abs_value(self) -> float | None:
        return self.p99_abs.value() if self.p99_abs is not None else None


@dataclass
class RunningPerDimRelStats:
    """按动作维度累计相对误差统计。"""

    rel_sum: list[float] | None = None
    rel_count: list[int] | None = None
    p99_rel: list[P2Quantile] | None = None

    def ensure_dim(self, dim: int) -> None:
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if self.rel_sum is not None and len(self.rel_sum) == d:
            return
        self.rel_sum = [0.0 for _ in range(d)]
        self.rel_count = [0 for _ in range(d)]
        self.p99_rel = [P2Quantile(0.99) for _ in range(d)]

    def update_rel_vec(self, rel_err_vec) -> None:
        if self.rel_sum is None or self.rel_count is None or self.p99_rel is None:
            self.ensure_dim(len(rel_err_vec))
        assert self.rel_sum is not None and self.rel_count is not None and self.p99_rel is not None
        if len(rel_err_vec) != len(self.rel_sum):
            # 若 action dim 变化，重置为新维度（通常不应发生）
            self.ensure_dim(len(rel_err_vec))
        for i, v in enumerate(rel_err_vec):
            fv = float(v)
            self.rel_sum[i] += fv
            self.rel_count[i] += 1
            self.p99_rel[i].update(fv)

    def mean_rel(self) -> list[float] | None:
        if self.rel_sum is None or self.rel_count is None:
            return None
        out: list[float] = []
        for s, c in zip(self.rel_sum, self.rel_count):
            out.append(float(s / float(c)) if c > 0 else float("nan"))
        return out

    def p99_rel_value(self) -> list[float] | None:
        if self.p99_rel is None:
            return None
        out: list[float] = []
        for q in self.p99_rel:
            v = q.value()
            out.append(float(v) if v is not None else float("nan"))
        return out


@dataclass
class RunningPerDimMsePctStats:
    """按动作维度累计 MSE 百分比（量化口径）。

    定义（累计到当前为止）：
        mse_pct_mean[d] = 100 * mean(diff[d]^2) / (mean(gt[d]^2) + eps)

    其中 mean 在时间维（step）上取平均。
    """

    eps: float = 1e-12
    err2_sum: list[float] | None = None
    gt2_sum: list[float] | None = None
    count: int = 0

    def ensure_dim(self, dim: int) -> None:
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if self.err2_sum is not None and len(self.err2_sum) == d:
            return
        self.err2_sum = [0.0 for _ in range(d)]
        self.gt2_sum = [0.0 for _ in range(d)]
        self.count = 0

    def update(self, diff_vec, gt_vec) -> None:
        if self.err2_sum is None or self.gt2_sum is None:
            self.ensure_dim(len(diff_vec))
        assert self.err2_sum is not None and self.gt2_sum is not None
        if len(diff_vec) != len(self.err2_sum):
            self.ensure_dim(len(diff_vec))
        for i, (dv, gv) in enumerate(zip(diff_vec, gt_vec)):
            d = float(dv)
            g = float(gv)
            self.err2_sum[i] += d * d
            self.gt2_sum[i] += g * g
        self.count += 1

    def mse_pct_mean(self) -> list[float] | None:
        if self.err2_sum is None or self.gt2_sum is None or self.count <= 0:
            return None
        out: list[float] = []
        c = float(self.count)
        for e2, g2 in zip(self.err2_sum, self.gt2_sum):
            mse = e2 / c
            g_pow = g2 / c
            out.append(float(100.0 * mse / (g_pow + float(self.eps))))
        return out

