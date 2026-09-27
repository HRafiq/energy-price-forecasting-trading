# Prior work on forecast-free benchmarks for storage

The forecast-free benchmark in `docs/results/t8_fixed_shape.md` is not a new idea.
This note records what was already known before I built it, what my numbers look
like beside the published ones, and which parts of my results I could not place in
the literature. Quotations from Sioshansi et al. and Le Roux-Tardif were read in
the body of the paper; those from Maciejowska et al. and Staffell and Rustomji
come from the abstract, which is marked where it applies.

## The idea is established, and it is old

**Sioshansi, Denholm, Jenkin and Weiss (2009).** "Estimating the value of
electricity storage in PJM: Arbitrage and some welfare effects", *Energy
Economics* 31(2), 269-277. DOI 10.1016/j.eneco.2008.10.005. They dispatch storage
on the previous two weeks' prices and settle it at actual prices, which they call
backcasting, and report the capture against perfect foresight for 2002-2007 in
their Figure 6, for a twelve-hour device. In their words the approach "represents a lower bound of value
capture that will almost certainly be enhanced by basic forecasting". They also
anticipate why the seasonal restriction matters here: "Although the diurnal hourly
price patterns differ significantly on a seasonal basis, such differences are
largely captured because our backcasting approach only uses a two-week lag." My
non-seasonal arm, which averages every trailing day, captures 68.7% against the
seasonal arm's 84.9%, which is the same effect from the other direction.

**Sioshansi, Denholm and Jenkin (2011).** "A comparative analysis of the value of
pure and hybrid electricity storage", *Energy Economics* 33(1), 56-66. DOI
10.1016/j.eneco.2010.06.004. The same technique at a one-week lag captures "at
least 89% of the theoretical perfect-foresight value of storage" for an eight-hour
pure storage device.

**Staffell and Rustomji (2016).** "Maximising the value of electricity storage",
*Journal of Energy Storage* 8, 212-225. DOI 10.1016/j.est.2016.08.010. From the abstract: "With no
foresight of future prices, 75-95% of the optimal profits are gained."

**Le Roux-Tardif (2026).** "Price Information Is Not Enough: Ordering and Decision
Rules in Storage Bidding", arXiv:2608.08377, August 2026. A preprint, not
peer-reviewed at the time of writing, and the closest thing to this experiment
that I found. Same power and energy as mine, 1 MW / 2 MWh, though at 85% round-trip against my 90% and 1.5 cycles a day
against my two, on 939 French day-ahead days: "The
perfect-foresight bidder earns 61.1 kEUR per megawatt-year of pure day-ahead
arbitrage. The climatological bidder, who knows only the average price for this
month and this hour, earns 47.5, or 78% of it." His recommendation is the one I
arrived at independently: "When reporting the value of forecasting, report it
against a climatological bidder rather than against zero."

One difference is worth stating because it runs in my favour. His month-hour mean
is estimated leave-one-out over his whole 31-month sample, so it draws on
same-month data from later years. Mine uses a strictly causal trailing window that
stops at the instant the delivery day begins, and `tests/test_t8_fixed_shape.py`
holds that boundary with a test that fails if one period of the target day leaks
in.

**Veenstra and Mulder (2025).** "Profitability of batteries in day-ahead and
intraday electricity markets: Assessment of operation strategies with endogenous
prices", *Energy Economics* 148, 108608. DOI
10.1016/j.eneco.2025.108608. The one study I found that puts a fixed schedule and
a naive forecast in the same framework, and it finds the opposite ordering to
mine: "the Predefined Periods strategy secures 58% of the profits achieved with
the Perfect Foresight strategy, while the Naive Forecast strategy captures even
66%." Their fixed rule is coarser than mine, being one set of buy and sell periods for
the whole year rather than a monthly average profile, and their study models price
impact endogenously and spans day-ahead and intraday, so several things differ at
once. I have not established which difference accounts for the reversal. It is a
genuine disagreement and not something to gloss over.

## Timing over accuracy was already published, on this market

**Maciejowska, Lipiecki and Uniejewski (2026).** "Statistical and economic
evaluation of forecasts in electricity markets: beyond RMSE and MAE", *Energy
Conversion and Management* 356, 121408. DOI 10.1016/j.enconman.2026.121408. On DE-LU, the same
market as this project, from the abstract:
"traditional accuracy metrics are only weakly correlated with BESS income", while
measures reflecting a forecast's "ability to reproduce daily price patterns" track
it far better. My own finding that an evening shifted one hour early costs as much
as noise with twice the error is the same result reached by a different route.

Le Roux-Tardif also proves the structural point that a battery returning to the
same state of charge each day is a spread instrument, so its schedule depends on
the ordering of periods and not on the price level, with magnitude entering only
through the threshold that round-trip efficiency imposes on a profitable spread.

## Where my numbers sit

No published range exists for the comparison I want, so this is a synthesis across
markets rather than a like-for-like band, and it should be read as such. The
forecast-free figures above are 78% (France), "at least 89%" for eight-hour storage
and 58% for a coarser fixed schedule (the Netherlands), with Staffell and Rustomji
quoting 75-95% across their cases. Mine, 84.9% on DE-LU, sits inside that scatter
rather than outside it. That is the main thing I wanted to know: the measurement is
ordinary, not anomalous. Markets, storage durations, cycle caps, wear costs and
periods all differ, so nothing here is a controlled comparison and no ranking
between these studies should be read into it.

## What I could not place in the literature

A search is weak evidence of absence, and these should be read as "not found in a
few hours of looking", not as claims of novelty:

* the per-day decomposition in `docs/results/t9_forecast_edge.md`, scoring each day
  by how far its ranking of periods departed from the seasonal shape and splitting
  the forecast's edge by quintile, including the result that the edge is
  indistinguishable from zero on the most ordinary fifth of days
* the finding that both arms degrade on atypical days but the forecast degrades
  about half as fast
* the regime test in `docs/results/t10_regime.md`: no robustness check of a
  forecast-free dispatch rule across the 2021-2023 gas crisis
* the seasonal against non-seasonal climatology comparison, 84.9% against 68.7%
