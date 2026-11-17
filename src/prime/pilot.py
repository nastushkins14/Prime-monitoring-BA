
import math
from statistics import NormalDist


def n_per_group(sigma, mde, alpha=0.05, power=0.8, comparisons=1):
    """Рассчитать необходимое количество клиентов в каждой группе."""
    z = NormalDist().inv_cdf

    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)

    n = 2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2
    return math.ceil(n)


def mde_for_n(sigma, n, alpha=0.05, power=0.8, comparisons=1):
    """Рассчитать минимальный обнаруживаемый эффект."""
    z = NormalDist().inv_cdf

    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)

    return (2 * (za + zb) ** 2 * sigma ** 2 / n) ** 0.5


def diff_ci(t, c, alpha=0.05):
    """Разность средних test − control, доверительный интервал и p-значение."""
    d = t.mean() - c.mean()

    # Стандартная ошибка разности средних
    se = (t.var() / len(t) + c.var() / len(c)) ** 0.5

    # Доверительный интервал и p-значение
    normal = NormalDist()
    q = normal.inv_cdf(1 - alpha / 2)
    p = 2 * (1 - normal.cdf(abs(d / se)))

    return {
        "diff": d,
        "lo": d - q * se,
        "hi": d + q * se,
        "p": p,
    }


def srm_pvalue(sizes):
    """Проверить соответствие размеров двух или трёх групп равным долям."""
    n = sum(sizes)
    expected = n / len(sizes)

    chi2 = sum((s - expected) ** 2 / expected for s in sizes)

    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))

    if len(sizes) == 3:
        return math.exp(-chi2 / 2)

    raise ValueError("Для четырёх и более групп используйте scipy.stats.chisquare")


def cuped(y, x):
    """Скорректировать метрику по предпериоду методом CUPED."""
    theta = y.cov(x) / x.var()

    y_cuped = y - theta * (x - x.mean())

    return y_cuped, theta

