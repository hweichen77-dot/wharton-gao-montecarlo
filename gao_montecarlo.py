"""Monte Carlo model for the 2026 Wharton Global Youth case study (Laura Gao / Creative Residency).

Timeline fixed by the case (Year 0 = 2026, all flows at the START of the year):
    2027 (Y1)  invest $300,000
    2028 (Y2)  invest $150,000
    2029-2032  no flows
    2033 (Y7)  split portfolio into Operating Reserve + Facility Contribution,
               then pay $50,000; repeat each year through 2042 (Y16), ten payments total.

Taxes and inflation indexing of the $50,000 payments are excluded, per the case.
"""

import argparse
import csv
import math
import os
from functools import lru_cache

import numpy as np

CONFIG = {
    "n_sims": 50_000,
    "seed": 20260920,

    "contributions": {2027: 300_000.0, 2028: 150_000.0},
    "split_year": 2033,
    "payment": 50_000.0,
    "n_payments": 10,

    # dist is "lognormal", "normal", or "bootstrap".
    # mean/std are ARITHMETIC simple annual returns. std is ignored under bootstrap,
    # which takes its spread and its year-to-year shape straight from history.
    # history_column only matters under bootstrap.

    # Phase 1: growth portfolio, start 2027 -> start 2033.
    "phase1": {"mean": 0.075, "std": 0.14, "dist": "lognormal",
               "history_column": "sp500_total_return"},

    # Phase 2: operating reserve (bond ladder), start 2033 -> start 2042.
    "phase2": {"mean": 0.035, "std": 0.045, "dist": "lognormal",
               "history_column": "tbond_10y_return"},

    # Realized annual returns, 1928-2025, from Damodaran's histretSP dataset.
    "history_csv": "data/sp500_annual_returns.csv",
    # Block length for the bootstrap. 1 resamples single years and throws away
    # serial structure; longer blocks keep runs of good and bad years intact,
    # which is what makes sequence-of-returns risk show up honestly.
    "bootstrap_block": 3,

    # Discount rate used to size the reserve as the PV of the ten payments.
    # Set it at or below the reserve's expected return: discounting at the full
    # expected return leaves zero cushion for a bad sequence of returns.
    "reserve_discount_rate": 0.030,

    "buffer_sweep": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "target_confidence": [0.95, 0.99],

    # Percentile band quoted to co-sponsors for the facility contribution.
    "credible_range_pct": (10, 90),

    # 2031 co-sponsor conversation. None -> use the simulated median 2031 value.
    "as_of_2031_value": None,
    # Buffer assumed when reporting facility figures outside the sweep.
    "base_buffer": 0.15,

    "sensitivity_phase1_means": [0.055, 0.065, 0.075, 0.085, 0.095],
    "sensitivity_phase2_means": [0.020, 0.030, 0.035, 0.040, 0.050],

    "save_charts": True,
    "chart_dir": ".",
}


# --- return generation -------------------------------------------------------

@lru_cache(maxsize=8)
def load_history(path, column):
    """Realized annual returns as a float array, in chronological order."""
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([float(r[column]) for r in rows])


def block_bootstrap(rng, shape, history, mean, block):
    """Resample contiguous blocks of realized returns, recentered on `mean`.

    Two things this buys over a fitted normal or lognormal. The empirical
    distribution carries real fat tails and skew instead of assumed ones, and
    sampling in blocks keeps consecutive years together, so a 2000-2002 style
    run can appear intact rather than being smoothed away by i.i.d. draws.

    Blocks wrap around the end of the series (circular bootstrap) so that every
    starting year is equally likely, including the last few.

    Recentering shifts the whole series by a constant so its mean equals `mean`,
    leaving spread and serial shape alone. The historical mean of the S&P is far
    above any defensible forward assumption, so the level has to be set by the
    analyst; the shape is what history is being asked for.
    """
    n_sims, n_years = shape
    h = history - history.mean() + mean
    n_blocks = -(-n_years // block)
    starts = rng.integers(0, len(h), size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(h)
    return h[idx.reshape(n_sims, -1)[:, :n_years]]


def phase_returns(rng, cfg, phase_key, shape, mean=None):
    """Draw returns for one phase, honoring that phase's distribution choice."""
    p = cfg[phase_key]
    m = p["mean"] if mean is None else mean
    if p["dist"] == "bootstrap":
        return block_bootstrap(rng, shape, load_history(cfg["history_csv"], p["history_column"]),
                               m, cfg["bootstrap_block"])
    return draw_returns(rng, shape, m, p["std"], p["dist"])


def draw_returns(rng, shape, mean, std, dist):
    """Simple annual returns, i.i.d.

    normal:    R ~ N(mean, std). Can produce R < -100%, so it is floored at -99%.
    lognormal: 1+R is lognormal with the SAME arithmetic mean and std. Matching
               moments: sigma^2 = ln(1 + std^2/(1+mean)^2), mu = ln(1+mean) - sigma^2/2.
               Preferred: returns stay above -100% and the compounding is right-skewed,
               which is how multi-year equity outcomes actually behave.
    """
    if dist == "normal":
        return np.maximum(rng.normal(mean, std, shape), -0.99)
    if dist == "lognormal":
        sigma2 = math.log(1.0 + (std ** 2) / ((1.0 + mean) ** 2))
        mu = math.log(1.0 + mean) - 0.5 * sigma2
        return np.exp(rng.normal(mu, math.sqrt(sigma2), shape)) - 1.0
    raise ValueError(f"unknown dist: {dist}")


# --- phase 1: accumulation ---------------------------------------------------

def simulate_accumulation(rng, cfg, phase1_mean=None):
    """Grow the two contributions to the start of every year 2027..2033.

    Returns an (n_sims, n_years+1) array of START-of-year portfolio values,
    indexed from 2027 through split_year. A contribution lands at the start of
    its year, so it earns that same year's return.
    """
    years = list(range(min(cfg["contributions"]), cfg["split_year"] + 1))
    rets = phase_returns(rng, cfg, "phase1", (cfg["n_sims"], len(years) - 1), phase1_mean)
    return years, accumulate_path(cfg, rets)


def accumulate_path(cfg, rets):
    """Apply an (n_paths, n_years) return matrix to the contribution schedule."""
    years = list(range(min(cfg["contributions"]), cfg["split_year"] + 1))
    path = np.zeros((rets.shape[0], len(years)))
    bal = np.zeros(rets.shape[0])
    for i, yr in enumerate(years):
        bal = bal + cfg["contributions"].get(yr, 0.0)
        path[:, i] = bal
        if i < len(years) - 1:
            bal = bal * (1.0 + rets[:, i])
    return path


def grow_forward(rng, values, n_years, cfg, mean=None):
    """Compound a set of starting values forward n_years with phase 1 returns."""
    rets = phase_returns(rng, cfg, "phase1", (values.shape[0], n_years), mean)
    return values * np.prod(1.0 + rets, axis=1)


# --- the 2033 split ----------------------------------------------------------

def pv_annuity_due(pmt, rate, n):
    """PV of n payments with the FIRST payment made immediately (annuity-due).

    Ordinary annuity: PMT * (1 - (1+r)^-n) / r  -> first payment one year out.
    The case pays at the START of each year, so every payment is one year earlier:
    multiply by (1+r). At r = 0 the PV is just n * PMT.
    """
    if abs(rate) < 1e-12:
        return pmt * n
    return pmt * (1.0 - (1.0 + rate) ** -n) / rate * (1.0 + rate)


def reserve_target(cfg, buffer):
    return pv_annuity_due(cfg["payment"], cfg["reserve_discount_rate"], cfg["n_payments"]) * (1.0 + buffer)


def simulate_reserve(rng, start_balances, cfg, phase2_mean=None):
    """Draw the reserve down by one payment at the start of each year.

    Order each year: pay first, THEN the surviving balance earns that year's return.
    This is where sequence-of-returns risk bites. Average return over the decade is
    not enough; a poor first few years shrinks the base that has to carry the rest,
    and there are no new contributions to repair it. A fixed payment against a
    shrinking balance means the withdrawal rate climbs every year a loss occurs.
    """
    rets = phase_returns(rng, cfg, "phase2",
                         (start_balances.shape[0], cfg["n_payments"]), phase2_mean)

    bal = start_balances.copy()
    ok = np.ones(start_balances.shape[0], dtype=bool)
    for t in range(cfg["n_payments"]):
        ok &= bal >= cfg["payment"] - 1e-6
        bal = np.maximum(bal - cfg["payment"], 0.0)
        if t < cfg["n_payments"] - 1:
            bal = bal * (1.0 + rets[:, t])
    return ok, bal


def split_at_2033(v2033, cfg, buffer):
    """Carve the reserve out first; the facility contribution is the residue.

    The reserve is a liability-matched set-aside, so it is funded ahead of the
    facility. If the portfolio cannot even cover the reserve, the facility
    contribution is zero and the reserve starts underfunded.
    """
    target = reserve_target(cfg, buffer)
    reserve = np.minimum(v2033, target)
    facility = np.maximum(v2033 - target, 0.0)
    return reserve, facility, target


def run_split(rng, v2033, cfg, buffer, phase2_mean=None):
    reserve, facility, target = split_at_2033(v2033, cfg, buffer)
    ok, ending = simulate_reserve(rng, reserve, cfg, phase2_mean)
    return {"buffer": buffer, "target": target, "success": ok.mean(),
            "facility": facility, "reserve": reserve, "ending": ending}


def required_buffer(rng_seed, v2033, cfg, confidence, phase2_mean=None, hi=3.0):
    """Smallest buffer reaching `confidence`, plus the ceiling that buffer can buy.

    Success is monotone in the buffer (more reserve never hurts), so bisect.
    Common random numbers: the same seed every evaluation, so differences across
    buffers come from the buffer, not from resampling noise.

    The ceiling matters. A buffer only reallocates the 2033 portfolio; it cannot
    create one. On paths where the whole portfolio is already below the reserve
    target, the payments fail no matter how the split is drawn. So the reachable
    success probability is bounded by P(portfolio 2033 >= reserve), and past that
    point more buffer only trades facility dollars for nothing. Returns
    (buffer or None, ceiling).
    """
    def prob(b):
        return run_split(np.random.default_rng(rng_seed), v2033, cfg, b, phase2_mean)["success"]

    ceiling = prob(hi)
    if ceiling < confidence:
        return None, ceiling
    lo = 0.0
    if prob(lo) >= confidence:
        return 0.0, ceiling
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if prob(mid) >= confidence:
            hi = mid
        else:
            lo = mid
    return hi, ceiling


# --- historical backtest -----------------------------------------------------

def rolling_windows(series, length):
    """Every overlapping window of `length` consecutive years, oldest first."""
    n = len(series) - length + 1
    return series[np.arange(n)[:, None] + np.arange(length)[None, :]]


def historical_backtest(cfg, band, buffer):
    """Replay actual history instead of sampling from a fitted distribution.

    Two passes. "as realized" uses the historical returns untouched, which is a
    fair question (what would this plan have done in the past) but a biased one
    for sizing, since the 1928-2025 S&P mean of about 11.9% is well above any
    forward assumption a team should defend. "recentered" shifts the series to
    the configured mean and keeps the historical spread and ordering, which is
    the real test of whether the quoted band covers realistic paths.

    Windows overlap, so these counts are not independent observations. Treat
    them as a sanity check on the model's shape, not as a coverage guarantee.
    """
    n_acc = cfg["split_year"] - min(cfg["contributions"])
    eq = load_history(cfg["history_csv"], cfg["phase1"]["history_column"])
    bd = load_history(cfg["history_csv"], cfg["phase2"]["history_column"])
    target = reserve_target(cfg, buffer)
    out = {}

    for label, shift_eq, shift_bd in [("as realized", 0.0, 0.0),
                                      ("recentered", cfg["phase1"]["mean"] - eq.mean(),
                                       cfg["phase2"]["mean"] - bd.mean())]:
        w_eq = rolling_windows(eq + shift_eq, n_acc)
        v = accumulate_path(cfg, w_eq)[:, -1]
        fac = np.maximum(v - target, 0.0)

        w_bd = rolling_windows(bd + shift_bd, cfg["n_payments"])
        # Fully funded reserve, real ten-year bond sequences: does the ladder hold?
        bal = np.full(w_bd.shape[0], target)
        ok = np.ones(w_bd.shape[0], dtype=bool)
        for t in range(cfg["n_payments"]):
            ok &= bal >= cfg["payment"] - 1e-6
            bal = np.maximum(bal - cfg["payment"], 0.0)
            if t < cfg["n_payments"] - 1:
                bal = bal * (1.0 + w_bd[:, t])

        out[label] = {
            "n_windows": len(v), "median_v": float(np.median(v)),
            "worst_v": float(v.min()), "best_v": float(v.max()),
            "in_band": float(((fac >= band[0]) & (fac <= band[1])).mean()),
            "below": float((fac < band[0]).mean()), "above": float((fac > band[1]).mean()),
            "reserve_ok": float(ok.mean()),
        }
    return out


# --- reporting ---------------------------------------------------------------

PCTS = [5, 25, 50, 75, 95]


def money(x):
    return f"${x:>12,.0f}"


def describe(a):
    return {"mean": a.mean(), **{f"p{p}": np.percentile(a, p) for p in PCTS}}


def print_dist(label, a):
    d = describe(a)
    print(f"  {label}")
    print(f"    mean   {money(d['mean'])}      median {money(d['p50'])}")
    print(f"    p5     {money(d['p5'])}      p25    {money(d['p25'])}")
    print(f"    p75    {money(d['p75'])}      p95    {money(d['p95'])}")


def main(cfg, charts=None):
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    n_acc = cfg["split_year"] - min(cfg["contributions"])

    print("=" * 78)
    print("LAURA GAO / CREATIVE RESIDENCY - MONTE CARLO")
    print("=" * 78)
    print(f"trials {cfg['n_sims']:,}   seed {seed}")
    for key, name in [("phase1", "phase 1 growth "), ("phase2", "phase 2 reserve")]:
        p = cfg[key]
        if p["dist"] == "bootstrap":
            h = load_history(cfg["history_csv"], p["history_column"])
            print(f"{name}  mean {p['mean']:.2%}  std {h.std(ddof=1):.2%} (historical)  "
                  f"bootstrap block {cfg['bootstrap_block']}y on {p['history_column']}")
        else:
            print(f"{name}  mean {p['mean']:.2%}  std {p['std']:.2%}  {p['dist']}")
    print(f"reserve discount rate {cfg['reserve_discount_rate']:.2%}")
    print(f"contributions {', '.join(f'{y}: ${v:,.0f}' for y, v in sorted(cfg['contributions'].items()))}")
    print(f"payments {cfg['n_payments']} x ${cfg['payment']:,.0f}, {cfg['split_year']}-{cfg['split_year'] + cfg['n_payments'] - 1}, annuity-due, not indexed")

    years, path = simulate_accumulation(rng, cfg)
    v2033 = path[:, -1]
    v2031 = path[:, years.index(2031)]

    print("\n" + "-" * 78)
    print(f"PHASE 1 - PORTFOLIO VALUE AT START OF {cfg['split_year']} ({n_acc} return years)")
    print("-" * 78)
    print(f"  total contributed ${sum(cfg['contributions'].values()):,.0f}")
    print_dist(f"value at start of {cfg['split_year']}", v2033)
    print(f"    P(loss vs contributed) {np.mean(v2033 < sum(cfg['contributions'].values())):.1%}")

    pv = pv_annuity_due(cfg["payment"], cfg["reserve_discount_rate"], cfg["n_payments"])
    print("\n" + "-" * 78)
    print("RESERVE SIZING - PV OF THE TEN PAYMENTS (ANNUITY-DUE)")
    print("-" * 78)
    print(f"  undiscounted total     {money(cfg['payment'] * cfg['n_payments'])}")
    print(f"  PV at {cfg['reserve_discount_rate']:.2%}            {money(pv)}   <- buffer 0%")

    print("\n" + "-" * 78)
    print("PAYMENT SUCCESS vs RESERVE BUFFER")
    print("-" * 78)
    print(f"  {'buffer':>7}  {'reserve set-aside':>18}  {'P(all 10 paid)':>15}  {'median facility':>16}  {'median left 2042':>17}")
    sweep = []
    for b in cfg["buffer_sweep"]:
        r = run_split(np.random.default_rng(seed + 1), v2033, cfg, b)
        sweep.append(r)
        print(f"  {b:>6.0%}  {money(r['target'])}      {r['success']:>13.2%}  "
              f"{money(np.median(r['facility']))}  {money(np.median(r['ending']))}")

    print("\n  buffer required to hit target confidence:")
    for c in cfg["target_confidence"]:
        b, ceiling = required_buffer(seed + 1, v2033, cfg, c)
        if b is None:
            print(f"    {c:.0%}: NOT REACHABLE at any buffer - ceiling is {ceiling:.1%}")
        else:
            print(f"    {c:.0%}: buffer {b:>6.1%}  -> reserve {money(reserve_target(cfg, b))}  (ceiling {ceiling:.1%})")
    cap = np.mean(v2033 >= reserve_target(cfg, 0.0))
    print(f"\n  P(2033 portfolio >= un-buffered reserve) = {cap:.1%}")
    print("  That is the hard ceiling on payment certainty. Buffer reallocates the 2033")
    print("  portfolio; it cannot create one. Raising certainty past the ceiling needs a")
    print("  lower phase 1 risk level (or a de-risking glidepath into 2033), not more buffer.")

    base_b = cfg["base_buffer"]
    base = run_split(np.random.default_rng(seed + 1), v2033, cfg, base_b)
    lo_p, hi_p = cfg["credible_range_pct"]
    print("\n" + "-" * 78)
    print(f"FACILITY CONTRIBUTION AT START OF {cfg['split_year']}  (buffer {base_b:.0%}, reserve {money(base['target'])})")
    print("-" * 78)
    print_dist("facility contribution", base["facility"])
    print(f"    P(facility = $0)        {np.mean(base['facility'] <= 0):.2%}")
    print(f"    p{lo_p}-p{hi_p} credible range  {money(np.percentile(base['facility'], lo_p))} to {money(np.percentile(base['facility'], hi_p))}")

    print("\n" + "-" * 78)
    print("2031 CO-SPONSOR VIEW - RANGE QUOTED TWO YEARS AHEAD")
    print("-" * 78)
    print_dist("portfolio at start of 2031 (unconditional)", v2031)
    anchor = cfg["as_of_2031_value"] or float(np.median(v2031))
    fwd = grow_forward(np.random.default_rng(seed + 2), np.full(cfg["n_sims"], anchor), 2, cfg)
    fac_fwd = np.maximum(fwd - base["target"], 0.0)
    print(f"  conditioning on a 2031 value of {money(anchor)}, two years of phase 1 returns:")
    print_dist(f"facility contribution in {cfg['split_year']}", fac_fwd)
    print(f"    p{lo_p}-p{hi_p} range to quote    {money(np.percentile(fac_fwd, lo_p))} to {money(np.percentile(fac_fwd, hi_p))}")
    print(f"    P(>= p{lo_p} figure)        {np.mean(fac_fwd >= np.percentile(fac_fwd, lo_p)):.0%}   "
          f"(quote the low end as the commitment, the high end as upside)")

    print("\n" + "-" * 78)
    print("SENSITIVITY - BUFFER NEEDED FOR EACH TARGET CONFIDENCE")
    print("-" * 78)
    for c in cfg["target_confidence"]:
        print(f"\n  target {c:.0%}      columns = phase 2 (reserve) mean return")
        print("  'cap NN%' = target unreachable at any buffer; NN% is the ceiling")
        hdr = "  p1 mean |" + "".join(f"{m:>9.1%}" for m in cfg["sensitivity_phase2_means"])
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for m1 in cfg["sensitivity_phase1_means"]:
            v = simulate_accumulation(np.random.default_rng(seed + 3), cfg, phase1_mean=m1)[1][:, -1]
            cells = []
            for m2 in cfg["sensitivity_phase2_means"]:
                b, ceiling = required_buffer(seed + 4, v, cfg, c, phase2_mean=m2)
                cells.append(f"cap {ceiling:>4.0%}" if b is None else f"{b:>8.1%}")
            print(f"  {m1:>6.1%}  |" + "".join(f"{c_:>9}" for c_ in cells))

    print(f"\n  facility contribution p{lo_p}-p{hi_p} (in $k) at buffer {base_b:.0%}")
    hdr = "  p1 mean |" + "".join(f"{m:>17.1%}" for m in cfg["sensitivity_phase2_means"])
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m1 in cfg["sensitivity_phase1_means"]:
        v = simulate_accumulation(np.random.default_rng(seed + 3), cfg, phase1_mean=m1)[1][:, -1]
        cells = []
        for m2 in cfg["sensitivity_phase2_means"]:
            f = run_split(np.random.default_rng(seed + 4), v, cfg, base_b, phase2_mean=m2)["facility"]
            cells.append(f"{np.percentile(f, lo_p) / 1e3:>6.0f}-{np.percentile(f, hi_p) / 1e3:<6.0f}")
        print(f"  {m1:>6.1%}  |" + "".join(f"{c_:>17}" for c_ in cells))
    print("  (facility barely moves with the phase 2 mean: the reserve is sized off the")
    print("   discount rate, not the realized reserve return, so phase 1 drives the range)")

    lo_band = float(np.percentile(base["facility"], lo_p))
    hi_band = float(np.percentile(base["facility"], hi_p))
    bt = historical_backtest(cfg, (lo_band, hi_band), base_b)
    print("\n" + "-" * 78)
    print(f"HISTORICAL BACKTEST - OVERLAPPING {n_acc}-YEAR WINDOWS, 1928-2025")
    print("-" * 78)
    print(f"  band being tested: p{lo_p}-p{hi_p} facility, {money(lo_band)} to {money(hi_band)}")
    for label, r in bt.items():
        print(f"\n  {label}  ({r['n_windows']} overlapping windows)")
        print(f"    {cfg['split_year']} value   median {money(r['median_v'])}  worst {money(r['worst_v'])}  best {money(r['best_v'])}")
        print(f"    facility inside band  {r['in_band']:>6.1%}   below {r['below']:.1%}   above {r['above']:.1%}")
        print(f"    reserve funded ten payments in {r['reserve_ok']:.1%} of real {cfg['n_payments']}-year bond sequences")
    print("\n  Windows overlap, so these are not independent trials. 'as realized' runs on the")
    print("  raw series, whose S&P mean is far above any forward assumption worth defending;")
    print("  'recentered' keeps the historical spread and ordering but sets the mean to the")
    print("  configured phase assumptions, which is the fairer test of the quoted band.")

    if cfg["save_charts"] if charts is None else charts:
        make_charts(cfg, v2033, base, sweep)
    print("\ndone.")


def make_charts(cfg, v2033, base, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = cfg["chart_dir"]
    for name, data, title, xlabel in [
        ("portfolio_2033.png", v2033, f"Portfolio value, start of {cfg['split_year']}", "value ($)"),
        ("facility_contribution.png", base["facility"],
         f"Facility contribution, start of {cfg['split_year']} (buffer {base['buffer']:.0%})", "value ($)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(data, bins=90, color="#1f4e79", alpha=0.85)
        for p, c in [(5, "#c00000"), (50, "#ffffff"), (95, "#c00000")]:
            ax.axvline(np.percentile(data, p), color=c, lw=1.4, ls="--",
                       label=f"p{p} ${np.percentile(data, p):,.0f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("trials")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{d}/{name}", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r["buffer"] for r in sweep], [r["success"] for r in sweep], marker="o", color="#1f4e79")
    for c in cfg["target_confidence"]:
        ax.axhline(c, ls="--", lw=1, color="#c00000")
        ax.annotate(f"{c:.0%}", (0, c), textcoords="offset points", xytext=(2, 4), color="#c00000")
    ax.set_title("Probability all ten $50,000 payments are made")
    ax.set_xlabel("reserve buffer over PV of payments")
    ax.set_ylabel("P(all 10 paid)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{d}/reserve_success_vs_buffer.png", dpi=140)
    plt.close(fig)
    print(f"\ncharts written to {d}/")


def selftest():
    assert abs(pv_annuity_due(50_000, 0.0, 10) - 500_000) < 1e-6
    assert abs(pv_annuity_due(100, 0.10, 1) - 100) < 1e-9
    assert abs(pv_annuity_due(100, 0.10, 2) - (100 + 100 / 1.10)) < 1e-9

    cfg = dict(CONFIG, n_sims=1000)
    cfg["phase1"] = {"mean": 0.08, "std": 0.0, "dist": "normal"}
    cfg["phase2"] = {"mean": 0.03, "std": 0.0, "dist": "normal"}
    cfg["reserve_discount_rate"] = 0.03

    v = simulate_accumulation(np.random.default_rng(0), cfg)[1][:, -1]
    expect = 300_000 * 1.08 ** 6 + 150_000 * 1.08 ** 5
    assert abs(v[0] - expect) < 1e-6, (v[0], expect)

    # Zero vol, reserve return == discount rate, no buffer: the ladder must fund
    # exactly ten payments and land on zero. This is the annuity identity.
    r = run_split(np.random.default_rng(0), v, cfg, 0.0)
    assert r["success"] == 1.0
    assert abs(r["ending"][0]) < 1e-6, r["ending"][0]

    # One dollar short of the PV must fail the tenth payment.
    short = np.full(5, reserve_target(cfg, 0.0) - 1.0)
    ok, _ = simulate_reserve(np.random.default_rng(0), short, cfg)
    assert not ok.any()

    # Bootstrap: draws come only from the (recentered) history, the recentering
    # hits the requested mean, and blocks stay contiguous in the source series.
    h = load_history(CONFIG["history_csv"], "sp500_total_return")
    assert len(h) > 50 and h.min() > -1.0
    b = block_bootstrap(np.random.default_rng(3), (5000, 6), h, 0.075, 3)
    assert abs(b.mean() - 0.075) < 0.01, b.mean()
    allowed = np.round(h - h.mean() + 0.075, 9)
    assert np.isin(np.round(b, 9), allowed).all()
    pairs = {(round(x, 9), round(y, 9)) for x, y in zip(allowed, allowed[1:])}
    pairs.add((round(allowed[-1], 9), round(allowed[0], 9)))
    assert all((round(b[i, j], 9), round(b[i, j + 1], 9)) in pairs
               for i in range(50) for j in (0, 3))

    cfg3 = dict(CONFIG, n_sims=2000)
    cfg3["phase1"] = dict(cfg3["phase1"], dist="bootstrap")
    cfg3["phase2"] = dict(cfg3["phase2"], dist="bootstrap")
    v3 = simulate_accumulation(np.random.default_rng(3), cfg3)[1][:, -1]
    assert v3.min() > 0 and np.isfinite(v3).all()
    bt = historical_backtest(cfg3, (0.0, 1e12), 0.15)
    assert bt["as realized"]["in_band"] == 1.0
    assert bt["recentered"]["n_windows"] == len(h) - 6 + 1

    # Lognormal draws preserve the arithmetic mean and keep returns above -100%.
    x = draw_returns(np.random.default_rng(1), 400_000, 0.075, 0.14, "lognormal")
    assert abs(x.mean() - 0.075) < 0.002 and abs(x.std() - 0.14) < 0.002
    assert x.min() > -1.0

    # Success probability is monotone in the buffer.
    cfg2 = dict(CONFIG, n_sims=4000)
    v2 = simulate_accumulation(np.random.default_rng(7), cfg2)[1][:, -1]
    probs = [run_split(np.random.default_rng(7), v2, cfg2, b)["success"] for b in (0.0, 0.1, 0.2, 0.3)]
    assert all(a <= b + 1e-12 for a, b in zip(probs, probs[1:])), probs
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sims", type=int)
    ap.add_argument("--dist", choices=["normal", "lognormal", "bootstrap"])
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--chart-dir")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        c = dict(CONFIG)
        if a.sims:
            c["n_sims"] = a.sims
        if a.dist:
            c["phase1"] = dict(c["phase1"], dist=a.dist)
            c["phase2"] = dict(c["phase2"], dist=a.dist)
        if a.chart_dir:
            c["chart_dir"] = a.chart_dir
        main(c, charts=False if a.no_charts else None)
