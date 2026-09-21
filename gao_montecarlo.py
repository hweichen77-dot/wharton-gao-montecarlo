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
import math

import numpy as np

CONFIG = {
    "n_sims": 50_000,
    "seed": 20260920,

    "contributions": {2027: 300_000.0, 2028: 150_000.0},
    "split_year": 2033,
    "payment": 50_000.0,
    "n_payments": 10,

    # Phase 1: growth portfolio, start 2027 -> start 2033.
    # mean/std are ARITHMETIC simple annual returns, i.i.d. across years.
    "phase1": {"mean": 0.075, "std": 0.14, "dist": "lognormal"},

    # Phase 2: operating reserve (bond ladder), start 2033 -> start 2042.
    "phase2": {"mean": 0.035, "std": 0.045, "dist": "lognormal"},

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
    p1 = cfg["phase1"]
    mean = p1["mean"] if phase1_mean is None else phase1_mean
    years = list(range(min(cfg["contributions"]), cfg["split_year"] + 1))
    rets = draw_returns(rng, (cfg["n_sims"], len(years) - 1), mean, p1["std"], p1["dist"])

    path = np.zeros((cfg["n_sims"], len(years)))
    bal = np.zeros(cfg["n_sims"])
    for i, yr in enumerate(years):
        bal = bal + cfg["contributions"].get(yr, 0.0)
        path[:, i] = bal
        if i < len(years) - 1:
            bal = bal * (1.0 + rets[:, i])
    return years, path


def grow_forward(rng, values, n_years, cfg, mean=None):
    """Compound a set of starting values forward n_years with phase 1 returns."""
    p1 = cfg["phase1"]
    rets = draw_returns(rng, (values.shape[0], n_years),
                        p1["mean"] if mean is None else mean, p1["std"], p1["dist"])
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
    p2 = cfg["phase2"]
    mean = p2["mean"] if phase2_mean is None else phase2_mean
    rets = draw_returns(rng, (start_balances.shape[0], cfg["n_payments"]),
                        mean, p2["std"], p2["dist"])

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
    print(f"phase 1 growth   mean {cfg['phase1']['mean']:.2%}  std {cfg['phase1']['std']:.2%}  {cfg['phase1']['dist']}")
    print(f"phase 2 reserve  mean {cfg['phase2']['mean']:.2%}  std {cfg['phase2']['std']:.2%}  {cfg['phase2']['dist']}")
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
    ap.add_argument("--dist", choices=["normal", "lognormal"])
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
