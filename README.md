# Gao Case Monte Carlo

Monte Carlo model for the 2026 Wharton Global Youth investment competition case study (Laura Gao / Creative Residency). How large does the 2033 operating reserve have to be before the ten $50,000 payments are safe, and how much is left over for the facility contribution?

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python gao_montecarlo.py
```

Flags:

```
--selftest              assertion checks (annuity identity, zero-vol exactness, monotonicity)
--sims N                trial count, default 50,000
--dist normal|lognormal|bootstrap  return distribution for both phases
--no-charts             skip the PNGs
--chart-dir DIR         where to write them
```

Every assumption lives in the `CONFIG` dict at the top of `gao_montecarlo.py`. Nothing below it needs editing to change a return assumption or a confidence target.

## Return assumptions

Three ways to generate annual returns, set per phase in `CONFIG` or for both at once with `--dist`.

`lognormal` and `normal` draw i.i.d. years from a fitted distribution using the mean and standard deviation you configure. Both accept per-year arrays, which is what the glidepath uses. `bootstrap` resamples contiguous blocks of realized annual returns from `data/sp500_annual_returns.csv`, recentered so the series mean equals your configured mean. Spread, skew, fat tails, and the ordering of good and bad years all come from history. The standard deviation setting is ignored in that mode.

The CSV holds S&P 500 total returns and 10-year Treasury returns for 1928 through 2025, extracted from Aswath Damodaran's `histretSP` dataset at NYU Stern. Realized S&P arithmetic mean over that span is 11.9% with a 19.4% standard deviation, well above what anyone should project forward, which is why the bootstrap recenters on your assumption instead of inheriting history's level.

Switching to bootstrap makes the case harder. Success ceiling drops from about 90% to about 80% and the odds of a zero facility contribution rise, because the historical spread is wider than a 14% assumption and bad years arrive in runs.

## De-risking glidepath

`phase1_glide` blends the growth assumption toward a conservative mix over the final years before the split. With the default `{"start_year": 2031, "end": {"mean": 0.05, "std": 0.08}}`, the 2027 through 2030 years keep 9% and 17%, 2031 runs at 7.7% and 14%, and 2032 at 6.3% and 11%. Set it to `None` for a static allocation.

Be careful what you claim for it. The glidepath moves the payment-success ceiling from 88.5% to 88.9%, which is almost nothing, because the early full-risk years carry both the most compounding and the most downside. It does tighten the 2031 co-sponsor range, and the report prints a static-allocation column beside the glidepath column so the comparison is visible rather than asserted.

## Two kinds of failure

The report splits payment failure into its two causes instead of reporting one number.

Phase 1 can end below the reserve target, in which case the set-aside was never funded and no split rule helps. Separately, a fully funded reserve can still run dry when weak returns arrive early in the payout decade, because a fixed $50,000 withdrawal against a shrinking balance is exposed to the order returns come in.

The buffer controls the second one and does nothing about the first. The report solves for both: the buffer that gets a funded reserve to your target confidence, and the buffer needed unconditionally, which at most allocations is unreachable at any size.

## What it models

Year 0 is 2026 and all cash flows happen at the start of a year, per the case.

- 2027, invest $300,000. 2028, add $150,000. Nothing else until 2033.
- Phase 1 grows both contributions to the start of 2033 with i.i.d. annual returns.
- At the start of 2033 the portfolio splits. The operating reserve is sized as the present value of ten $50,000 payments as an annuity-due, scaled by a configurable buffer. Whatever exceeds it becomes the facility contribution.
- Phase 2 draws the reserve down one payment at a time, start of year, with the surviving balance earning that year's return. Paying before the return is what exposes sequence-of-returns risk.

Taxes and inflation indexing of the payments are excluded, which the case allows.

## Outputs

Console report covering the 2033 portfolio distribution, payment success rate across a 0% to 30% buffer sweep, the buffer needed for a 95% or 99% target, the facility contribution distribution with a percentile band for co-sponsors, a 2031 forward-looking version of that band, and sensitivity tables over Phase 1 and Phase 2 return assumptions.

The script also writes three PNGs to the working directory. One histogram of 2033 portfolio value, one of the facility contribution, and a line chart of success probability against buffer size.

## Historical backtest

The report replays every overlapping six-year window from 1928 to 2025 through the same contribution schedule, and every overlapping ten-year window through the reserve drawdown. It does this twice. Once on the raw series, and once with the series recentered on the configured means, which is the fairer test of whether the quoted percentile band covers realistic paths.

Overlapping windows are not independent trials. Read this as a sanity check on the model's shape, not as a coverage guarantee.

## The success ceiling

At the default assumptions, payment success flattens near 89% and no buffer gets past it. Roughly one path in ten finishes 2033 below the reserve target itself, so the split has nothing to work with. A buffer moves money between the two pools, it cannot add money to the portfolio.

The allocation tradeoff table prices what buying past that ceiling costs. Reaching 95% unconditionally needs a 35/65 mix, which cuts the median facility contribution from about 190k to about 28k. Ten payments of $50,000 consume most of what $450,000 can safely produce in six years, so the honest answer is to quote 95% on the funded reserve, state the portfolio-size risk separately, and show the tradeoff rather than claim a certainty the arithmetic does not support.
