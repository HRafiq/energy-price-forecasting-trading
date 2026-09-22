# Bid curves from the quantile fan: a fixed plan

Written 2026-09-19, before any code for it existed, and not changed after the run
starts. Committed on its own before the code.

## The question, in plain words

Today the battery bids like a price taker: for each quarter-hour it submits a
volume and accepts whatever price the auction clears at. A real desk submits a
price-quantity curve instead: "sell up to 1 MW, but only if the price is at least
X" and "buy up to 1 MW, but only if the price is at most Y". Orders that the
clearing price does not reach simply do not execute. The forecast already carries
the natural limits: its lower quantiles say what a bad price for a sale looks like,
its upper quantiles what a bad price for a purchase looks like. The T5 note names
the risk this guards against: a fixed-volume strategy can buy into a surprise
spike. Does bidding with limits from the fan earn more than bidding volumes?

## What a limit does to the day

An order that does not execute changes the battery's day. A withheld purchase
means less energy stored, so a later sale the plan counted on may not be
deliverable in full; a withheld sale leaves energy in the store. The plan does not
pretend these away:

* The day is replayed period by period. An executed purchase adds energy; an
  executed sale draws it. If a sale executes but the store cannot cover it, the
  shortfall is a delivery failure, settled at the German imbalance price for that
  quarter-hour, as the T4 outage experiment settles deviations (a short position
  pays the reBAP price). Withheld orders themselves cost nothing: nothing was
  committed.
* At the end of the day the store may sit above or below the level every day
  starts and ends at. That gap is valued as T4 values it, at the day's terminal
  prices, so energy left in the store counts for what it could have fetched and a
  deficit counts against.
* The planned volumes are the production schedule, unchanged: the optimiser is not
  told about the limits. Making it plan for them (linked or block orders, or a
  stochastic plan over the fan) is a further step this experiment does not take.

## The arms

All on the production model's saved validation forecasts, the 1 MW / 2 MWh
battery, €8 wear, two-cycle cap, the same 730 days:

* **Control, `volumes`:** median dispatch, fixed volumes at the clearing price, the
  backtest as it stands. It must reproduce the saved validation profit to the cent.
* **`limits_q25`:** the same volumes, each sale with limit q25 and each purchase
  with limit q75. A sale executes if the realised price is at least q25; a purchase
  if it is at most q75. Wide enough that most orders execute.
* **`limits_q10`:** the same, with limits q10 and q90. Only prices far outside the
  fan withhold an order.

Both limit arms settle exactly as the control where every order executes.

## The criterion, fixed before the run

For each limit arm separately: adopt it only if its mean daily profit difference
against `volumes`, imbalance and end-of-day gap included, over the 730 validation
days has a 95% moving-block bootstrap interval (7-day blocks, 5,000 draws) entirely
above zero. If both pass, the one with the higher mean is preferred and the
difference between them is reported.

## Reported alongside, not part of the decision

How many orders each arm withheld, split into sales and purchases; how many were
withheld on days the price spiked; profit on spike days against other days; the
imbalance cash and the end-of-day gap value per arm; the June to September and
October to May split; the difference with imbalance and gap costs left out, to show
how much of the change is the withheld orders themselves.

## Risks named in advance

1. **Withheld purchases starve the evening.** A purchase withheld at midday because
   the price ran above q75 may be the very energy the evening sale needed; the
   imbalance charge for that shortfall can outweigh the purchase avoided.
2. **Withheld sales on a spike day.** A sale withheld because the price fell below
   q25 keeps energy that then earns nothing until the day ends; the end-of-day
   value may not cover it.
3. **The fan is not calibrated for this.** On validation the 90% range covers about
   85% of prices, so q10 and q90 limits withhold more than 10% of orders each way.
4. **Imbalance prices are extreme.** reBAP reaches thousands of euros per MWh on
   rare quarter-hours; a single shortfall on such a quarter-hour can dominate a
   month. The bootstrap interval will show it, and the spike-day split is reported.

## What each outcome means

* An arm's interval above zero: adopt it, push, production note, and limit orders
  become a candidate for the live plan step (a plan carries a limit per period).
* Neither: reported as tested and rejected, with what the withheld orders cost and
  saved, and the T5 note updated with the measured answer.
