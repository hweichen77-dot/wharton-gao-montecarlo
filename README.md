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
--dist normal|lognormal return distribution for both phases
--no-charts             skip the PNGs
--chart-dir DIR         where to write them
```

Every assumption lives in the `CONFIG` dict at the top of `gao_montecarlo.py`. Nothing below it needs editing to change a return assumption or a confidence target.

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

## The success ceiling

At the default assumptions, payment success flattens near 90% and no buffer gets past it. Roughly one path in ten finishes 2033 below the reserve target itself, so the split has nothing to work with. A buffer moves money between the two pools, it cannot add money to the portfolio. Reaching 95% means lowering Phase 1 risk or de-risking into 2033, not setting aside more.
