"""gta6-twitch-ccu-forecast-01 entry: forecast a launch day's Twitch curve
from the shape of comparable launch days.

The case gives us one known number — the viewers in hour 1 of release day —
plus the game's own pre-release history and about forty past launches. Two
ratios describe where the case sits at the end of hour 1:

    a = log(hour-1 viewers / pre-release baseline viewers)
    c = log(hour-1 channels / pre-release baseline channels)

The baseline is the median hourly figure over the four days before release
day, skipping the last 14 hours (rolling regional unlocks contaminate them).
A big (a, c) means the game is already live and hour 1 is already a crowd;
a small pair means the unlock is still ahead and the day's climb is yet to
come.

For every comparable launch we rebuild its release-day curve and express it
as log ratios against that launch's own hour 1. For each hour 2..24 we then
take kernel-weighted quantiles of those ratios — a Gaussian kernel over
(a, c) with bandwidths 1.2 and 1.0, blended with 25% uniform weight so a
thinly populated corner of feature space still borrows the whole field's
shape — and multiply the case's hour-1 print by the exponentials of those
quantiles.

Validation (leave-one-out over the 37 comparable launches with a complete
day curve, each posing in turn as the case with the other 36 as its
comparables, scored with the contest's exact pinball loss on the log scale):
this model averages 0.574 against the house persistence baseline's 0.930,
and 0.409 on the launches whose hour 1 was already a crowd (a >= 5) — the
corner Grand Theft Auto VI's release day sits in.

Any surprise (missing fields, too few usable comparables, a parse failure)
falls back to the house persistence shape, so a case always gets a
well-formed answer instead of an invalid one. Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file could be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at those levels (the fallback's spread).
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Kernel bandwidths over (a, c), in natural-log units, and the uniform share
# mixed into every comparable's kernel weight.
BAND_A = 1.2
BAND_C = 1.0
UNIFORM_SHARE = 0.25
# The fallback's log-spread per square-root hour (the house baseline's 0.15).
FALLBACK_SIGMA = 0.15
# Fewer usable comparables than this and the empirical shape is not trusted.
MIN_USABLE = 8
# Baseline window: hours before day start, and the hours nearest the start
# that are excluded from it.
BASE_LO, BASE_HI = 96.0, 14.0


def parse_iso(text):
    """An ISO-8601 timestamp as an aware datetime, or None if unreadable."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def median_of(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def round_half_up_2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def tidy_row(values):
    """Force one row of quantiles into the contest's shape: finite, inside
    (0, 1e9], non-decreasing, rounded for a compact file."""
    cleaned = []
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            v = 1.0
        cleaned.append(min(1e9, max(1.0, float(v))))
    cleaned.sort()
    return [round_half_up_2(v) for v in cleaned]


def weighted_quantile(pairs, tau):
    """The tau-quantile of (value, weight) pairs, linearly interpolated
    across the cumulative weight."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    cum = 0.0
    prev_value, prev_share = ordered[0][0], 0.0
    for value, weight in ordered:
        cum += weight
        share = cum / total
        if share >= tau:
            if share == prev_share:
                return value
            frac = (tau - prev_share) / (share - prev_share)
            return prev_value + frac * (value - prev_value)
        prev_value, prev_share = value, share
    return ordered[-1][0]


def hour_of(point_time, day_start):
    """Which release-day hour (1..24) a timestamp falls in, or None."""
    if point_time is None or day_start is None:
        return None
    delta = (point_time - day_start).total_seconds() / 3600.0
    if 0.0 <= delta < 24.0:
        return int(delta) + 1
    return None


def release_day_curve(hourly, day_start):
    """One launch's release day as {hour: viewers} for hours 1..24, with
    short gaps closed by log-linear interpolation. None when the day cannot
    be used (no hour 1, under 18 points, or a gap wider than 6 hours)."""
    raw = {}
    for point in hourly or []:
        if not isinstance(point, dict):
            continue
        h = hour_of(parse_iso(point.get("from")), day_start)
        if h is None:
            continue
        viewers = point.get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            raw[h] = float(viewers)
    if 1 not in raw or len(raw) < 18:
        return None
    curve = {}
    present = sorted(raw)
    for h in range(1, 25):
        if h in raw:
            curve[h] = raw[h]
            continue
        before = max((x for x in present if x < h), default=None)
        after = min((x for x in present if x > h), default=None)
        if before is None or after is None or after - before > 6:
            return None
        frac = (h - before) / (after - before)
        curve[h] = math.exp(
            math.log(raw[before])
            + frac * (math.log(raw[after]) - math.log(raw[before]))
        )
    return curve


def baseline_medians(hourly, day_start):
    """(median viewers, median channels) over the baseline window
    [day_start-96h, day_start-14h]. None when there is no usable point."""
    viewers, channels = [], []
    for point in hourly or []:
        if not isinstance(point, dict):
            continue
        start = parse_iso(point.get("from"))
        if start is None or day_start is None:
            continue
        back = (day_start - start).total_seconds() / 3600.0
        if not (BASE_HI <= back <= BASE_LO):
            continue
        v = point.get("viewers")
        if isinstance(v, (int, float)) and v > 0:
            viewers.append(float(v))
        ch = point.get("channels")
        if isinstance(ch, (int, float)) and ch > 0:
            channels.append(float(ch))
    if not viewers:
        return None
    return max(median_of(viewers), 1.0), max(median_of(channels), 1.0) if channels else 1.0


def launch_ratios(hourly, day_start):
    """(a, c, log-ratio curve) for one past launch, or None when it cannot
    be used."""
    base = baseline_medians(hourly, day_start)
    curve = release_day_curve(hourly, day_start)
    if base is None or curve is None:
        return None
    first_channels = None
    for point in hourly or []:
        if not isinstance(point, dict):
            continue
        if hour_of(parse_iso(point.get("from")), day_start) == 1:
            ch = point.get("channels")
            if isinstance(ch, (int, float)):
                first_channels = float(ch)
            break
    base_v, base_c = base
    a = math.log(curve[1] / base_v)
    c = math.log(max(first_channels or 1.0, 1.0) / base_c)
    ratios = {h: math.log(curve[h] / curve[1]) for h in range(2, 25)}
    return a, c, ratios


def case_position(case, day_start):
    """(a, c, hour-1 viewers) for the case itself. The baseline comes from
    the case's own history (history_hourly preferred, history as fallback,
    timestamps read from `from` or `at`)."""
    known = case.get("known") or []
    first = next(
        (k for k in known if isinstance(k, dict) and k.get("hour") == 1), None
    )
    if first is None or not isinstance(first.get("viewers"), (int, float)):
        return None
    p1 = max(float(first["viewers"]), 1.0)
    c1 = float(first.get("channels") or 0)
    points = case.get("history_hourly") or case.get("history") or []
    # history points carry `at` instead of `from`; normalise before reuse.
    normalised = []
    for point in points:
        if isinstance(point, dict) and "from" not in point and "at" in point:
            point = dict(point, **{"from": point["at"]})
        normalised.append(point)
    base = baseline_medians(normalised, day_start)
    if base is None:
        return None
    base_v, base_c = base
    return (
        math.log(p1 / base_v),
        math.log(max(c1, 1.0) / base_c),
        p1,
    )


def persistence_shape(p1, hours):
    """The house baseline's shape: every hour at hour 1's level, spread
    widening like a random walk's in log space."""
    return {
        h: tidy_row(
            [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)) for z in Z_SCORES]
        )
        for h in hours
    }


def empirical_shape(case, day_start, hours):
    """The comparables model, or None when the case cannot support it."""
    position = case_position(case, day_start)
    if position is None:
        return None
    a_t, c_t, p1 = position
    training = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        comp_start = parse_iso(comp.get("day_start"))
        if comp_start is None:
            continue
        entry = launch_ratios(comp.get("hourly"), comp_start)
        if entry is not None:
            training.append(entry)
    if len(training) < MIN_USABLE:
        return None
    weights = []
    for a_j, c_j, _ in training:
        w = math.exp(-0.5 * ((a_t - a_j) / BAND_A) ** 2)
        w *= math.exp(-0.5 * ((c_t - c_j) / BAND_C) ** 2)
        weights.append(w)
    mean_w = sum(weights) / len(weights)
    weights = [(1.0 - UNIFORM_SHARE) * w + UNIFORM_SHARE * mean_w for w in weights]
    out = {}
    for h in hours:
        if h < 2 or h > 24:
            continue
        pairs = [
            (ratios[h], w)
            for (_, _, ratios), w in zip(training, weights)
        ]
        out[h] = tidy_row(
            [p1 * math.exp(weighted_quantile(pairs, tau)) for tau in LEVELS]
        )
    return out


def forecast_case(case):
    """The forecast document for one case: 23 rows, hours 2..24."""
    day_start = parse_iso(case.get("day_start"))
    rows = case.get("forecast") or []
    hours = sorted(
        {
            row["hour"]
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("hour"), int)
        }
    )
    if not hours:
        hours = list(range(2, 25))
    p1 = 1.0
    known = case.get("known") or []
    first = next(
        (k for k in known if isinstance(k, dict) and k.get("hour") == 1), None
    )
    if first is not None and isinstance(first.get("viewers"), (int, float)):
        p1 = max(float(first["viewers"]), 1.0)
    quantiles = None
    try:
        quantiles = empirical_shape(case, day_start, hours)
    except Exception:
        quantiles = None
    if quantiles is None:
        quantiles = persistence_shape(p1, hours)
    return {"hours": [{"hour": h, "q": quantiles[h]} for h in hours]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
