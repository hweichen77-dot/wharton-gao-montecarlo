
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

    "phase1": {"mean": 0.09, "std": 0.17, "dist": "lognormal",
               "history_column": "sp500_total_return"},

    "phase2": {"mean": 0.03, "std": 0.045, "dist": "lognormal",
               "history_column": "tbond_10y_return"},

    "phase1_glide": {"start_year": 2031, "end": {"mean": 0.05, "std": 0.08}},

    "history_csv": "data/sp500_annual_returns.csv",
    "bootstrap_block": 3,

    "reserve_discount_rate": 0.030,

    "buffer_sweep": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "target_confidence": [0.95, 0.99],

    "credible_range_pct": (20, 80),

    "as_of_2031_value": None,
    "base_buffer": 0.13,

    "allocation_grid": [("85/15", 0.090, 0.170), ("75/25", 0.082, 0.145),
                        ("65/35", 0.074, 0.125), ("55/45", 0.066, 0.105),
                        ("45/55", 0.058, 0.085), ("35/65", 0.050, 0.070)],

    "sensitivity_phase1_means": [0.055, 0.065, 0.075, 0.085, 0.095],
    "sensitivity_phase2_means": [0.020, 0.030, 0.035, 0.040, 0.050],

    "save_charts": True,
    "chart_dir": ".",
}


# reads the old return numbers out of the csv file
@lru_cache(maxsize=8)
def load_history(path, column):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([float(r[column]) for r in rows])


# grabs random chunks of real history so bad years stay stuck together
def block_bootstrap(rng, shape, history, mean, block):
    n_sims, n_years = shape
    h = history - history.mean() + mean
    n_blocks = -(-n_years // block)
    starts = rng.integers(0, len(h), size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(h)
    return h[idx.reshape(n_sims, -1)[:, :n_years]]


# makes the portfolio play it safer each year as 2033 gets closer
def glide_schedule(cfg, start_year, n_years, mean, std):
    g = cfg.get("phase1_glide")
    years = np.arange(start_year, start_year + n_years)
    if not g:
        return np.full(n_years, mean), np.full(n_years, std)
    span = cfg["split_year"] - g["start_year"] + 1
    t = np.clip((years - g["start_year"] + 1) / span, 0.0, 1.0)
    return mean + t * (g["end"]["mean"] - mean), std + t * (g["end"]["std"] - std)


# gets the yearly returns for one phase of the plan
def phase_returns(rng, cfg, phase_key, shape, mean=None, start_year=None):
    p = cfg[phase_key]
    m = p["mean"] if mean is None else mean
    if p["dist"] == "bootstrap":
        return block_bootstrap(rng, shape, load_history(cfg["history_csv"], p["history_column"]),
                               m, cfg["bootstrap_block"])
    if phase_key == "phase1" and start_year is not None:
        m, sd = glide_schedule(cfg, start_year, shape[1], m, p["std"])
        return draw_returns(rng, shape, m, sd, p["dist"])
    return draw_returns(rng, shape, m, p["std"], p["dist"])


# rolls the dice for a bunch of random yearly returns
def draw_returns(rng, shape, mean, std, dist):
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if dist == "normal":
        return np.maximum(rng.normal(mean, std, shape), -0.99)
    if dist == "lognormal":
        sigma2 = np.log(1.0 + (std ** 2) / ((1.0 + mean) ** 2))
        mu = np.log(1.0 + mean) - 0.5 * sigma2
        return np.exp(rng.normal(mu, np.sqrt(sigma2), shape)) - 1.0
    raise ValueError(f"unknown dist: {dist}")


# grows the two donations year by year up to the 2033 split
def simulate_accumulation(rng, cfg, phase1_mean=None):
    years = list(range(min(cfg["contributions"]), cfg["split_year"] + 1))
    rets = phase_returns(rng, cfg, "phase1", (cfg["n_sims"], len(years) - 1), phase1_mean,
                         start_year=years[0])
    return years, accumulate_path(cfg, rets)


# adds the money in each year then multiplies by that year's return
def accumulate_path(cfg, rets):
    years = list(range(min(cfg["contributions"]), cfg["split_year"] + 1))
    path = np.zeros((rets.shape[0], len(years)))
    bal = np.zeros(rets.shape[0])
    for i, yr in enumerate(years):
        bal = bal + cfg["contributions"].get(yr, 0.0)
        path[:, i] = bal
        if i < len(years) - 1:
            bal = bal * (1.0 + rets[:, i])
    return path


# takes some starting money and grows it a few more years
def grow_forward(rng, values, n_years, cfg, mean=None, start_year=None):
    rets = phase_returns(rng, cfg, "phase1", (values.shape[0], n_years), mean,
                         start_year=start_year)
    return values * np.prod(1.0 + rets, axis=1)


# how much you need today to cover payments starting right now
def pv_annuity_due(pmt, rate, n):
    if abs(rate) < 1e-12:
        return pmt * n
    return pmt * (1.0 - (1.0 + rate) ** -n) / rate * (1.0 + rate)


# the reserve size once you add the safety buffer on top
def reserve_target(cfg, buffer):
    return pv_annuity_due(cfg["payment"], cfg["reserve_discount_rate"], cfg["n_payments"]) * (1.0 + buffer)


# pays out 50k a year and checks if the money lasts ten years
def simulate_reserve(rng, start_balances, cfg, phase2_mean=None):
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


# fills the reserve first, whatever is left goes to the building
def split_at_2033(v2033, cfg, buffer):
    target = reserve_target(cfg, buffer)
    reserve = np.minimum(v2033, target)
    facility = np.maximum(v2033 - target, 0.0)
    return reserve, facility, target


# does the split then checks if all ten payments went through
def run_split(rng, v2033, cfg, buffer, phase2_mean=None):
    reserve, facility, target = split_at_2033(v2033, cfg, buffer)
    ok, ending = simulate_reserve(rng, reserve, cfg, phase2_mean)
    return {"buffer": buffer, "target": target, "success": ok.mean(),
            "facility": facility, "reserve": reserve, "ending": ending}


# the odds of paying everyone if the reserve started with exactly enough
def funded_success(rng, cfg, buffer, phase2_mean=None):
    ok, _ = simulate_reserve(rng, np.full(cfg["n_sims"], reserve_target(cfg, buffer)),
                             cfg, phase2_mean)
    return float(ok.mean())


# keeps guessing buffer sizes until the funded reserve is safe enough
def required_buffer_funded(rng_seed, cfg, confidence, hi=1.0):
    def prob(b):
        return funded_success(np.random.default_rng(rng_seed), cfg, b)

    if prob(hi) < confidence:
        return None
    lo = 0.0
    if prob(lo) >= confidence:
        return 0.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if prob(mid) >= confidence:
            hi = mid
        else:
            lo = mid
    return hi


# same buffer search but counting the times phase 1 came up short
def required_buffer(rng_seed, v2033, cfg, confidence, phase2_mean=None, hi=3.0):
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


# chops the history into every possible run of back to back years
def rolling_windows(series, length):
    n = len(series) - length + 1
    return series[np.arange(n)[:, None] + np.arange(length)[None, :]]


# replays what actually happened instead of made up numbers
def historical_backtest(cfg, band, buffer):
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


PCTS = [5, 25, 50, 75, 95]


# prints a number so it looks like dollars
def money(x):
    return f"${x:>12,.0f}"


# grabs the average and a few percentiles
def describe(a):
    return {"mean": a.mean(), **{f"p{p}": np.percentile(a, p) for p in PCTS}}


# prints the spread of numbers in a block you can read
def print_dist(label, a):
    d = describe(a)
    print(f"  {label}")
    print(f"    mean   {money(d['mean'])}      median {money(d['p50'])}")
    print(f"    p5     {money(d['p5'])}      p25    {money(d['p25'])}")
    print(f"    p75    {money(d['p75'])}      p95    {money(d['p95'])}")


# runs the whole thing and prints the report
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
    g = cfg.get("phase1_glide")
    if g:
        print(f"phase 1 glidepath  from {g['start_year']} toward mean {g['end']['mean']:.2%} "
              f"std {g['end']['std']:.2%} at {cfg['split_year']}")
    else:
        print("phase 1 glidepath  none (static allocation)")
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

    no_glide = dict(cfg, phase1_glide=None)
    v_ng = simulate_accumulation(np.random.default_rng(seed), no_glide)[1][:, -1]

    print("\n" + "-" * 78)
    print("PAYMENT SUCCESS vs RESERVE BUFFER")
    print("-" * 78)
    print(f"  {'buffer':>7}  {'reserve':>12}  {'glide on':>9}  {'glide off':>10}  "
          f"{'funded only':>12}  {'med facility':>13}  {'med left 2042':>14}")
    sweep, sweep_ng = [], []
    for b in cfg["buffer_sweep"]:
        r = run_split(np.random.default_rng(seed + 1), v2033, cfg, b)
        r_ng = run_split(np.random.default_rng(seed + 1), v_ng, no_glide, b)
        sweep.append(r)
        sweep_ng.append(r_ng)
        fs = funded_success(np.random.default_rng(seed + 1), cfg, b)
        print(f"  {b:>6.0%}  {r['target']:>12,.0f}  {r['success']:>8.1%}  {r_ng['success']:>9.1%}  "
              f"{fs:>11.1%}  {np.median(r['facility']):>13,.0f}  {np.median(r['ending']):>14,.0f}")
    print("  'funded only' = P(all 10 paid) conditional on the reserve starting fully funded.")

    print("\n  buffer required for the FUNDED reserve to hit target confidence:")
    for c in cfg["target_confidence"]:
        bf = required_buffer_funded(seed + 1, cfg, c)
        print(f"    {c:.0%}: buffer {bf:>6.1%}  -> reserve {money(reserve_target(cfg, bf))}")
    print("  This is the operating-commitment guarantee. It is the number to quote for the")
    print("  ten payments, because it is the one the reserve design actually controls.")

    print("\n  buffer required for UNCONDITIONAL success (includes phase 1 shortfall):")
    for c in cfg["target_confidence"]:
        b, ceiling = required_buffer(seed + 1, v2033, cfg, c)
        b_ng, ceil_ng = required_buffer(seed + 1, v_ng, no_glide, c)
        fmt = lambda x, cl: f"NOT REACHABLE (ceiling {cl:.1%})" if x is None else f"buffer {x:.1%}"
        print(f"    {c:.0%}  glide on: {fmt(b, ceiling):<32} glide off: {fmt(b_ng, ceil_ng)}")
        if b is not None:
            print(f"          -> reserve {money(reserve_target(cfg, b))}")

    print("\n" + "-" * 78)
    print(f"FAILURE DECOMPOSITION (buffer {cfg['base_buffer']:.0%})")
    print("-" * 78)
    tgt = reserve_target(cfg, cfg["base_buffer"])
    underfunded = float(np.mean(v2033 < tgt))
    fs = funded_success(np.random.default_rng(seed + 1), cfg, cfg["base_buffer"])
    total = run_split(np.random.default_rng(seed + 1), v2033, cfg, cfg["base_buffer"])["success"]
    print(f"  P(2033 value < reserve of {money(tgt).strip()})   {underfunded:>7.1%}   phase 1 never funded the set-aside")
    print(f"  P(ladder fails | reserve fully funded)    {1 - fs:>7.1%}   sequence risk inside the payout decade")
    print(f"  P(all ten payments made)                  {total:>7.1%}")
    print("\n  These are two different failures with two different fixes. Ladder failure is a")
    print("  sequence problem inside the payout decade and the buffer controls it directly.")
    print("  Underfunding is a phase 1 risk-level problem, and the buffer does nothing for")
    print("  it. Quoting the first number as the funding risk leaves the second unstated.")

    cap = np.mean(v2033 >= reserve_target(cfg, 0.0))
    print(f"\n  P(2033 portfolio >= un-buffered reserve) = {cap:.1%}")
    print("  That is the hard ceiling on payment certainty. Buffer reallocates the 2033")
    print("  portfolio; it cannot create one. The glidepath trims this only slightly,")
    print("  because the early full-risk years carry the most compounding and the most")
    print("  downside. Moving the ceiling means a lower risk level throughout, which the")
    print("  allocation tradeoff table below prices in facility dollars.")

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
    fwd = grow_forward(np.random.default_rng(seed + 2), np.full(cfg["n_sims"], anchor), 2, cfg,
                       start_year=2031)
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

    print("\n" + "-" * 78)
    print("ALLOCATION TRADEOFF - WHAT UNCONDITIONAL CERTAINTY ACTUALLY COSTS")
    print("-" * 78)
    print(f"  {'mix':>16} {'mean/std':>11} {'ceiling':>8} {'buf@%.0f%%' % (cfg['target_confidence'][0] * 100):>9} "
          f"{'med V2033':>11} {'med facility':>13}")
    for lbl, m, sd in cfg["allocation_grid"]:
        alt = dict(cfg, phase1_glide=None, phase1=dict(cfg["phase1"], mean=m, std=sd))
        va = simulate_accumulation(np.random.default_rng(seed + 5), alt)[1][:, -1]
        b, ceiling = required_buffer(seed + 6, va, alt, cfg["target_confidence"][0])
        fac = run_split(np.random.default_rng(seed + 6), va, alt, b if b is not None else base_b)["facility"]
        print(f"  {lbl:>16} {m:>5.1%}/{sd:>4.0%} {ceiling:>7.1%} "
              f"{('%.1f%%' % (b * 100)) if b is not None else 'n/a':>9} "
              f"{np.median(va):>11,.0f} {np.median(fac):>13,.0f}")
    print("\n  Unconditional certainty is bought almost entirely out of the facility")
    print("  contribution. The ten payments consume most of what 450,000 of contributions")
    print("  can safely produce in six years, so a portfolio conservative enough to make")
    print("  them near-certain has little surplus left to give the residency.")

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
        funded = [funded_success(np.random.default_rng(seed + 1), cfg, b) for b in cfg["buffer_sweep"]]
        make_charts(cfg, v2033, base, sweep, sweep_ng, funded)
    print("\ndone.")


# draws the graphs and saves them as pngs
def make_charts(cfg, v2033, base, sweep, sweep_ng=None, funded=None):
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
    ax.plot([r["buffer"] for r in sweep], [r["success"] for r in sweep], marker="o",
            color="#1f4e79", label="with glidepath")
    if sweep_ng:
        ax.plot([r["buffer"] for r in sweep_ng], [r["success"] for r in sweep_ng], marker="s",
                color="#888888", ls="--", label="static allocation")
    if funded:
        ax.plot(cfg["buffer_sweep"], funded, marker="^", color="#2e7d32",
                label="reserve funded (operating commitment only)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.4, 1.02)
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


# checks the math against answers we can work out by hand
def selftest():
    assert abs(pv_annuity_due(50_000, 0.0, 10) - 500_000) < 1e-6
    assert abs(pv_annuity_due(100, 0.10, 1) - 100) < 1e-9
    assert abs(pv_annuity_due(100, 0.10, 2) - (100 + 100 / 1.10)) < 1e-9

    cfg = dict(CONFIG, n_sims=1000, phase1_glide=None)
    cfg["phase1"] = {"mean": 0.08, "std": 0.0, "dist": "normal"}
    cfg["phase2"] = {"mean": 0.03, "std": 0.0, "dist": "normal"}
    cfg["reserve_discount_rate"] = 0.03

    v = simulate_accumulation(np.random.default_rng(0), cfg)[1][:, -1]
    expect = 300_000 * 1.08 ** 6 + 150_000 * 1.08 ** 5
    assert abs(v[0] - expect) < 1e-6, (v[0], expect)

    gz = dict(cfg, phase1_glide={"start_year": 2031, "end": {"mean": 0.05, "std": 0.0}})
    gm = glide_schedule(gz, 2027, 6, 0.08, 0.0)[0]
    vg = simulate_accumulation(np.random.default_rng(0), gz)[1][:, -1]
    expect_g = 300_000 * np.prod(1 + gm) + 150_000 * np.prod(1 + gm[1:])
    assert abs(vg[0] - expect_g) < 1e-6, (vg[0], expect_g)

    r = run_split(np.random.default_rng(0), v, cfg, 0.0)
    assert r["success"] == 1.0
    assert abs(r["ending"][0]) < 1e-6, r["ending"][0]

    short = np.full(5, reserve_target(cfg, 0.0) - 1.0)
    ok, _ = simulate_reserve(np.random.default_rng(0), short, cfg)
    assert not ok.any()

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

    gcfg = dict(CONFIG, n_sims=20_000)
    gm, gs = glide_schedule(gcfg, 2027, 6, 0.09, 0.17)
    assert np.allclose(gm, [0.09, 0.09, 0.09, 0.09, 0.09 - 0.04 / 3, 0.09 - 0.08 / 3])
    assert np.allclose(gs, [0.17, 0.17, 0.17, 0.17, 0.14, 0.11])
    assert np.allclose(glide_schedule(gcfg, 2031, 2, 0.09, 0.17)[1], [0.14, 0.11])
    assert np.allclose(glide_schedule(dict(gcfg, phase1_glide=None), 2027, 6, 0.09, 0.17)[0], 0.09)

    mv = np.array([0.02, 0.09, 0.15])
    x = draw_returns(np.random.default_rng(5), (300_000, 3), mv, np.array([0.05, 0.1, 0.2]), "lognormal")
    assert np.allclose(x.mean(axis=0), mv, atol=0.003), x.mean(axis=0)

    v_g = simulate_accumulation(np.random.default_rng(6), gcfg)[1][:, -1]
    v_s = simulate_accumulation(np.random.default_rng(6), dict(gcfg, phase1_glide=None))[1][:, -1]
    assert v_g.std() < v_s.std(), (v_g.std(), v_s.std())

    for b in (0.0, 0.1, 0.2):
        uncond = run_split(np.random.default_rng(6), v_g, gcfg, b)["success"]
        assert funded_success(np.random.default_rng(6), gcfg, b) >= uncond - 1e-9
    fs = [funded_success(np.random.default_rng(6), gcfg, b) for b in (0.0, 0.1, 0.2, 0.3)]
    assert all(a <= b + 1e-12 for a, b in zip(fs, fs[1:])), fs
    assert abs(funded_success(np.random.default_rng(6), gcfg,
                              required_buffer_funded(6, gcfg, 0.95)) - 0.95) < 0.01

    x = draw_returns(np.random.default_rng(1), 400_000, 0.075, 0.14, "lognormal")
    assert abs(x.mean() - 0.075) < 0.002 and abs(x.std() - 0.14) < 0.002
    assert x.min() > -1.0

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
