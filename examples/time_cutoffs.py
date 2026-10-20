"""Point-in-time predictions over rows that carry times.

A query with a cutoff reads only rows from before it, which lets us backtest by
predicting each day from the days before it. We then show a table that refits on
its own once its columns drift.

Run with:
  uv run python examples/time_cutoffs.py
"""

import numpy as np

import generalforecast

DAYS, PER_DAY = 30, 40
generator = np.random.default_rng(0)

# Halfway through, the sign of column 0 flips, so older days mislead.
days = np.repeat(np.arange(DAYS), PER_DAY)
X = generator.normal(size=(len(days), 4)).astype(np.float32)
drift = np.where(days < DAYS // 2, 1.0, -1.0)
y = (drift * X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int32)

client = generalforecast.local()
table = client.create_table(
    "events",
    model="general-forecast/tabicl-v2-classifier",
    features=4,
    classes=2,
    strategy=generalforecast.Strategy.neighbors(256),
    # Pages never straddle a day, so earlier days keep their caches.
    time_bucket=1,
)
table.insert(np.arange(len(X)), X, y, times=days)

for day in range(5, DAYS, 5):
    today = days == day
    predicted = table.predict(X[today], before=day).argmax(axis=1)
    print(
        f"day {day:2}: accuracy from earlier days {np.mean(predicted == y[today]):.2f}"
    )

everything = table.predict(X[days == 20]).argmax(axis=1)
print(f"day 20 reading every day: {np.mean(everything == y[days == 20]):.2f}")

# This table refits once a fifth of its rows have changed, or a column's mean
# moves by more than its standard deviation.
live = client.create_table(
    "live",
    model="general-forecast/tabicl-v2-classifier",
    features=4,
    classes=2,
    refit_after=0.2,
    refit_shift=1.0,
)
live.insert(np.arange(400), X[:400], y[:400])
live.predict(X[:5])
shifted = X[400:500] + 3.0
live.insert(np.arange(400, 500), shifted, y[400:500])
live.predict(shifted[:5])
print("the live table refit on its own after its columns shifted")
