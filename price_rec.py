import numpy as np
from db_cleaning_map import COUNTRY_TO_REGION


def to_scalar(x):
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return float(x)


def exp_safe(v):
    return float(np.exp(np.clip(v, -700, 700)))


def is_missing_category(value):
    return value is None or value == "Other"


def format_vnd(value):
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if not np.isfinite(value):
            return "N/A"
        return f"{int(round(value)):,} VND"
    except Exception:
        return str(value)


def format_range_vnd(value_range):
    if value_range is None:
        return "N/A"
    low, high = value_range
    if low is None or high is None:
        return "N/A"
    return f"{format_vnd(low)} to {format_vnd(high)}"


def format_pct(value, decimals=1):
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if not np.isfinite(value):
            return "N/A"
        return f"{value * 100:.{decimals}f}%"
    except Exception:
        return "N/A"


def reliability_label(n):
    if n is None or n == 0:
        return "No Data"
    if n < 5:
        return "Very Low"
    if n < 20:
        return "Moderate"
    return "High"


def diversity_label(unique_n):
    if unique_n is None or unique_n == 0:
        return "No Variation"
    if unique_n == 1:
        return "Single Price Level"
    if unique_n <= 2:
        return "Very Low Variation"
    if unique_n <= 4:
        return "Low Variation"
    return "Normal Variation"


def model_quality_label(pred_metrics, global_median_price):
    if pred_metrics is None or global_median_price is None or global_median_price <= 0:
        return "Unknown"

    mae_k = pred_metrics["MAE"]  
    cov50 = pred_metrics["Coverage_50"]
    cov90 = pred_metrics["Coverage_90"]

    score = 0

    if mae_k is not None and np.isfinite(mae_k):
        mae_vnd = mae_k * 1000
        rel_mae = mae_vnd / global_median_price
        if rel_mae <= 0.10:
            score += 2
        elif rel_mae <= 0.20:
            score += 1

    if cov50 is not None and np.isfinite(cov50):
        if 0.40 <= cov50 <= 0.60:
            score += 1

    if cov90 is not None and np.isfinite(cov90):
        if 0.80 <= cov90 <= 0.98:
            score += 2
        elif 0.70 <= cov90 < 0.80:
            score += 1

    if score >= 4:
        return "Good"
    if score >= 2:
        return "Reasonable"
    return "Weak"


def get_quantiles_df(
    df,
    target,
    map=None,
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    min_samples=5,
):
    filtered_df = df.copy()

    if map is not None:
        for key, values in map.items():
            if values is None:
                continue
            if isinstance(values, list):
                if len(values) == 0:
                    filtered_df = filtered_df.iloc[0:0]
                else:
                    filtered_df = filtered_df[filtered_df[key].isin(values)]
            else:
                filtered_df = filtered_df[filtered_df[key] == values]

    series = filtered_df[target].dropna()

    if len(series) == 0:
        return {
            "n": 0,
            "unique_n": 0,
            "min": None,
            "max": None,
            "quantiles": None,
        }

    n = int(len(series))
    unique_n = int(series.nunique())
    min_val = float(series.min())
    max_val = float(series.max())

    if n < min_samples:
        return {
            "n": n,
            "unique_n": unique_n,
            "min": min_val,
            "max": max_val,
            "quantiles": None,
        }

    q = series.quantile(list(quantiles)).to_dict()
    q = {float(k): float(v) for k, v in q.items()}

    return {
        "n": n,
        "unique_n": unique_n,
        "min": min_val,
        "max": max_val,
        "quantiles": q,
    }


def get_competitors(df, compete, map=None):
    filtered_df = df.copy()

    if map is not None:
        for key, values in map.items():
            if values is None:
                continue
            if isinstance(values, list):
                if len(values) == 0:
                    filtered_df = filtered_df.iloc[0:0]
                else:
                    filtered_df = filtered_df[filtered_df[key].isin(values)]
            else:
                filtered_df = filtered_df[filtered_df[key] == values]

    return filtered_df[compete].dropna().unique().tolist()


def get_price_quantiles(pred_dist, quantile_levels):
    quantiles_vnd = {}
    for q in quantile_levels:
        log_val = to_scalar(pred_dist.ppf(q))
        val = exp_safe(log_val) * 1000
        quantiles_vnd[float(q)] = float(val)

    mu_log = to_scalar(getattr(pred_dist, "mu", np.nan))
    sigma_log = to_scalar(getattr(pred_dist, "sigma", np.nan))

    if np.isfinite(mu_log) and np.isfinite(sigma_log):
        mean = np.exp(mu_log + 0.5 * sigma_log**2) * 1000
        std = np.sqrt((np.exp(sigma_log**2) - 1.0) * np.exp(2 * mu_log + sigma_log**2)) * 1000
    else:
        mean = np.nan
        std = np.nan

    return {
        "mean": float(mean) if np.isfinite(mean) else None,
        "std": float(std) if np.isfinite(std) else None,
        "quantiles": quantiles_vnd,
    }


def _expected_profit_curve(pred_dist, quantity, cost, n_grid=800):
    y_low = to_scalar(pred_dist.ppf(0.001))
    y_high = to_scalar(pred_dist.ppf(0.999))

    if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low >= y_high:
        y_low = to_scalar(pred_dist.ppf(0.1))
        y_high = to_scalar(pred_dist.ppf(0.9))
        if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low >= y_high:
            y_low, y_high = -1.0, 1.0

    y_grid = np.linspace(y_low, y_high, n_grid)
    x_vals = np.exp(np.clip(y_grid, -700, 700)) * 1000

    try:
        cdf_vals = np.asarray(pred_dist.cdf(y_grid), dtype=float).reshape(-1)
        if cdf_vals.size != x_vals.size:
            raise ValueError("Unexpected cdf output shape")
    except Exception:
        cdf_vals = np.array([to_scalar(pred_dist.cdf(y)) for y in y_grid], dtype=float)

    survival_vals = np.clip(1.0 - cdf_vals, 0.0, 1.0)
    expected_profit = (x_vals * quantity - cost) * survival_vals
    expected_profit = np.maximum(expected_profit, 0.0)

    valid = np.isfinite(x_vals) & np.isfinite(expected_profit)
    return x_vals[valid], expected_profit[valid]


def summarize_expected_profit(pred_dist, quantity, cost, pct_of_max=0.8):
    x_vals, expected_profit = _expected_profit_curve(pred_dist, quantity, cost)

    if len(x_vals) == 0:
        return {
            "max_expected_profit": None,
            "optimal_price": None,
            "price_range_ge_pct_max": None,
        }

    max_idx = int(np.argmax(expected_profit))
    max_profit = float(expected_profit[max_idx])
    optimal_price = float(x_vals[max_idx])

    threshold = pct_of_max * max_profit
    mask = expected_profit >= threshold

    price_range = None
    if mask.any():
        xs = x_vals[mask]
        price_range = (float(xs.min()), float(xs.max()))

    return {
        "max_expected_profit": max_profit,
        "optimal_price": optimal_price,
        "price_range_ge_pct_max": price_range,
    }


def constrained_profit_summary(pred_dist, quantity, cost, allowed_range, pct_of_local_max=0.9, n_grid=800):
    if allowed_range is None:
        return {
            "optimal_price": None,
            "max_expected_profit": None,
            "price_range_ge_pct_local_max": None,
        }

    x_vals, expected_profit = _expected_profit_curve(pred_dist, quantity, cost, n_grid=n_grid)
    if len(x_vals) == 0:
        return {
            "optimal_price": None,
            "max_expected_profit": None,
            "price_range_ge_pct_local_max": None,
        }

    low, high = allowed_range
    mask = (x_vals >= low) & (x_vals <= high)

    if not mask.any():
        return {
            "optimal_price": None,
            "max_expected_profit": None,
            "price_range_ge_pct_local_max": None,
        }

    xs = x_vals[mask]
    profits = expected_profit[mask]

    max_idx = int(np.argmax(profits))
    local_max_profit = float(profits[max_idx])
    local_optimal_price = float(xs[max_idx])

    threshold = pct_of_local_max * local_max_profit
    range_mask = profits >= threshold

    local_range = None
    if range_mask.any():
        xr = xs[range_mask]
        local_range = (float(xr.min()), float(xr.max()))

    return {
        "optimal_price": local_optimal_price,
        "max_expected_profit": local_max_profit,
        "price_range_ge_pct_local_max": local_range,
    }


def build_band_from_stats(qdict, min_samples=5):
    """
    Returns:
        {
            "band": (low, high) or None,
            "band_type": "p25_p75" | "p10_p90" | "min_max" | "point" | "none",
            "anchor": float or None,
        }
    """
    if qdict is None or qdict["n"] == 0:
        return {"band": None, "band_type": "none", "anchor": None}

    q = qdict.get("quantiles")
    min_val = qdict.get("min")
    max_val = qdict.get("max")
    unique_n = qdict.get("unique_n", 0)

    # If there is only one unique price, treat it as a point anchor immediately.
    if unique_n == 1 and min_val is not None:
        return {"band": None, "band_type": "point", "anchor": float(min_val)}

    # If sample size is enough and price variation is meaningful, prefer percentile bands.
    if qdict["n"] >= min_samples and q is not None and unique_n >= 5:
        if 0.25 in q and 0.75 in q:
            low, high = float(q[0.25]), float(q[0.75])
            if low < high:
                return {"band": (low, high), "band_type": "p25_p75", "anchor": None}

        if 0.10 in q and 0.90 in q:
            low, high = float(q[0.10]), float(q[0.90])
            if low < high:
                return {"band": (low, high), "band_type": "p10_p90", "anchor": None}

    # If there are only a few unique price levels, keep the segment but downgrade the band.
    if min_val is not None and max_val is not None:
        min_val, max_val = float(min_val), float(max_val)
        if min_val < max_val:
            return {"band": (min_val, max_val), "band_type": "min_max", "anchor": None}
        return {"band": None, "band_type": "point", "anchor": min_val}

    return {"band": None, "band_type": "none", "anchor": None}


def intersect_ranges(r1, r2):
    if r1 is None or r2 is None:
        return None
    low = max(float(r1[0]), float(r2[0]))
    high = min(float(r1[1]), float(r2[1]))
    if low > high:
        return None
    return (low, high)


def narrow_band_around_point(point, width_pct=0.05):
    if point is None:
        return None
    try:
        point = float(point)
        if not np.isfinite(point) or point <= 0:
            return None
        low = point * (1 - width_pct)
        high = point * (1 + width_pct)
        if low < high:
            return (low, high)
    except Exception:
        return None
    return None


def get_global_median_price(df):
    series = df["unit_price"].dropna()
    if len(series) == 0:
        return None
    return float(series.median())


def build_logic_steps(
    segment_source,
    segment_band,
    segment_n,
    segment_unique_n,
    global_profit_band,
    intersection_band,
    method,
    user_config,
    segment_band_type="p25_p75",
    segment_anchor=None,
):  
    region_of_origin = COUNTRY_TO_REGION.get(user_config["country_of_origin"], "Other")
    source_map = {
        "competitor_same_manufacturer": f"We first identified contractors that have previously won bids from {user_config["investor"]}, then reviewed their winning prices for products from {user_config["manufacturer"]}.",
        "competitor_same_country": f"We first identified contractors that have previously won bids from {user_config["investor"]}, then reviewed their winning prices for products from {user_config["country_of_origin"]}.",
        "competitor_same_region": f"We first identified contractors that have previously won bids from {user_config["investor"]}, then reviewed their winning prices for products from {region_of_origin}.",
        "competitor": f"We first identified contractors that have previously won bids from {user_config["investor"]}, then reviewed their winning prices for this product.",
        "global": f"We first reviewed all previous winning prices for this product.",
        None: "We used the available historical data as the reference point.",
    }

    steps = [
        source_map.get(segment_source, source_map[None]),
        f"This sample contains {segment_n} observations of winning prices. From this sample, we build the historical reference pricing range by considering price percentiles.",
    ]

    if segment_band_type == "p25_p75":
        steps.append(
            f"We used the 25th - 75th percentile (middle 50% band) as the reference range. This band is {format_range_vnd(segment_band)}."
        )
    elif segment_band_type == "p10_p90":
        steps.append(
            f"The 25th - 75th percentile (middle 50% band) was too narrow, so we used the 10th - 90th percentile as the reference range. This band is {format_range_vnd(segment_band)}."
        )
    elif segment_band_type == "min_max":
        steps.append(
            f"These prices are concentrated in only a small number of repeated levels, so we used the entire price span from min to max ({format_range_vnd(segment_band)}) as the reference range rather than using percentiles."
        )
    elif segment_band_type == "point":
        steps.append(
            f"The prices are concentrated at a single repeated price point of {format_vnd(segment_anchor)}, so no range could be formed and we use that point as the fallback for recommendation."
        )
        return steps

    steps.append(
        f"We then consider the model's prediction and identify a price range that preserves at least 80% of the maximum expected profit. That model-supported range is {format_range_vnd(global_profit_band)}."
    )

    if method == "intersection_of_segment_and_global_profit_band":
        steps.append(
            f"We intersected the historical reference range with the model-supported range. The overlap is {format_range_vnd(intersection_band)}."
        )
        steps.append(
            "The **recommended range** is thus the **full overlap**, and the **recommended optimal price** is the **price inside that overlap** that gives **the highest expected profit** (as predicted by the model)."
        )
    else:
        steps.append(
            "The historical reference range does not overlap the model-supported range."
        )
        steps.append(
            "The **recommended optimal price** is thus the price **within the historical reference range** that gives the **highest expected profit** as predicted by the model."
        )
        steps.append(
            "The **recommended range** includes prices within the **historical reference range** that still **preserve at least 90% of the maximum expected profit** achievable inside that range."
        )

    return steps


def build_risk_text(segment_n, segment_unique_n, pred_metrics, segment_source, global_median_price):
    hist_conf = reliability_label(segment_n)
    model_quality = model_quality_label(pred_metrics, global_median_price)

    risk_parts = [
        f"The historical reference has {hist_conf.lower()} confidence level.\n"
    ]

    if pred_metrics is not None:
        mae_k = pred_metrics.get("MAE")  # thousand VND
        cov50 = pred_metrics.get("Coverage_50")
        cov90 = pred_metrics.get("Coverage_90")

        mae_vnd = None
        rel_mae = None
        if mae_k is not None and np.isfinite(mae_k):
            mae_vnd = mae_k * 1000

        if (
            mae_vnd is not None
            and global_median_price is not None
            and np.isfinite(global_median_price)
            and global_median_price > 0
        ):
            rel_mae = mae_vnd / global_median_price

        risk_parts.append(
            f"- The model quality is {model_quality.lower()}.\n"
        )
    else:
        risk_parts.append(
            "- Model quality could not be evaluated because prediction metrics were not provided.\n"
        )

    if segment_source in {"competitor", "global"}:
        risk_parts.append(
            "- Because the selected historical reference is broad, the recommendation may be less specific to your exact product segment."
        )

    return " ".join(risk_parts)


def build_recommendation_text(rec):
    if rec["status"] != "ok":
        return (
            "Recommendation\n"
            "- We could not generate a reliable unit price recommendation from the available data.\n\n"
            "Why\n"
            "- The system could not find a stable and relevant historical price reference with enough information to support a recommendation.\n"
            "- Without a stable historical reference, any specific price suggestion would be too uncertain.\n\n"
            "Reliability\n"
            "- Please review this bid manually or use a broader internal pricing rule before submitting a price."
        )

    why_lines = "\n".join([f"- {step}" for step in rec["logic_steps"]])

    return (
        "Recommendation\n"
        f"- Recommended optimal price: **{format_vnd(rec['recommended_price'])}** per unit.\n"
        f"- Recommended price range: **{format_range_vnd(rec['recommended_range'])}** per unit.\n\n"
        "Why\n"
        "- TL;DR: We combine **historical winning prices from the most relevant market segment** and **machine learning-based predictions**.\n"
        f"{why_lines}\n\n"
        "Reliability\n"
        f"- {rec['risk_text']}"
    )


def select_segment_band(
    df,
    user_config,
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    min_samples=5,
):
    region_of_origin = COUNTRY_TO_REGION.get(user_config["country_of_origin"], "Other")

    investor_missing = is_missing_category(user_config.get("investor"))
    manu_missing = is_missing_category(user_config.get("manufacturer"))
    coo_missing = is_missing_category(user_config.get("country_of_origin"))
    ror_missing = is_missing_category(region_of_origin)

    if investor_missing:
        competitors = []
    else:
        competitors = get_competitors(
            df,
            "contractor_name",
            {"investor": user_config["investor"]},
        )

    candidates = []

    if len(competitors) > 0 and not manu_missing:
        q = get_quantiles_df(
            df,
            "unit_price",
            map={"contractor_name": competitors, "manufacturer": user_config["manufacturer"]},
            quantiles=quantiles,
            min_samples=min_samples,
        )
        candidates.append(("competitor_same_manufacturer", q))

    if len(competitors) > 0 and not coo_missing:
        q = get_quantiles_df(
            df,
            "unit_price",
            map={"contractor_name": competitors, "country_of_origin": user_config["country_of_origin"]},
            quantiles=quantiles,
            min_samples=min_samples,
        )
        candidates.append(("competitor_same_country", q))

    if len(competitors) > 0 and not ror_missing:
        q = get_quantiles_df(
            df,
            "unit_price",
            map={"contractor_name": competitors, "region_of_origin": region_of_origin},
            quantiles=quantiles,
            min_samples=min_samples,
        )
        candidates.append(("competitor_same_region", q))

    if len(competitors) > 0:
        q = get_quantiles_df(
            df,
            "unit_price",
            map={"contractor_name": competitors},
            quantiles=quantiles,
            min_samples=min_samples,
        )
        candidates.append(("competitor", q))

    q_global = get_quantiles_df(
        df,
        "unit_price",
        map=None,
        quantiles=quantiles,
        min_samples=min_samples,
    )
    candidates.append(("global", q_global))

    for source, stats in candidates:
        if stats is None:
            continue

        # Hard rule: never use a segment with population below min_samples.
        if stats.get("n", 0) < min_samples:
            continue

        band_info = build_band_from_stats(stats, min_samples=min_samples)
        if band_info["band"] is not None or band_info["band_type"] == "point":
            return {
                "competitors": competitors,
                "region_of_origin": region_of_origin,
                "selected_source": source,
                "selected_stats": stats,
                "selected_band": band_info["band"],
                "selected_band_type": band_info["band_type"],
                "selected_anchor": band_info["anchor"],
                "candidates": candidates,
                "global_stats": q_global,
            }

    return {
        "competitors": competitors,
        "region_of_origin": region_of_origin,
        "selected_source": None,
        "selected_stats": None,
        "selected_band": None,
        "selected_band_type": "none",
        "selected_anchor": None,
        "candidates": candidates,
        "global_stats": q_global,
    }


def recommend_price(
    df,
    user_config,
    pred_dist,
    pred_metrics,
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    min_samples=5,
    global_profit_pct=0.8,
    local_profit_pct=0.9,
):
    """
    Rule-based pricing engine.

    Logic:
    1. Build fallback hierarchy:
       competitor + same manufacturer
       > competitor + same country of origin
       > competitor + same region of origin
       > competitor
       > global

    2. Take the first segment with a usable historical reference.
       Preferred reference = p25-p75.
       Safeguards:
       - if p25-p75 collapses or price variation is too limited, widen or downgrade
       - if distinct price levels are very few, use min-max as a cautious span
       - if all observations share one repeated price, treat it as a point anchor

    3. Build the model-supported good-profit band:
       expected profit >= global_profit_pct * global max expected profit

    4. If the historical reference overlaps with the model-supported band:
       - recommended range = overlap
       - recommended price = best expected-profit point inside overlap

    5. If there is no overlap:
       - optimize expected profit inside the historical reference only
       - recommended range = prices inside that reference with profit >= local_profit_pct * local max

    6. If only a point anchor exists:
       - use a narrow soft band around the anchor for optimization
       - if that still fails, return the anchor as a conservative fallback
    """
    quantity = user_config.get("quantity")
    cost = user_config.get("cost")

    if quantity is None or cost is None:
        raise ValueError("user_config must contain non-null 'quantity' and 'cost'")

    segment_info = select_segment_band(
        df=df,
        user_config=user_config,
        quantiles=quantiles,
        min_samples=min_samples,
    )

    segment_source = segment_info["selected_source"]
    segment_stats = segment_info["selected_stats"]
    segment_band = segment_info["selected_band"]
    segment_band_type = segment_info["selected_band_type"]
    segment_anchor = segment_info["selected_anchor"]

    global_median_price = get_global_median_price(df)

    if segment_stats is None:
        rec = {
            "status": "no_reliable_segment_band",
            "segment_source": None,
            "segment_stats": None,
            "segment_band": None,
            "segment_band_type": "none",
            "segment_anchor": None,
            "profit_band_global": None,
            "intersection_band": None,
            "recommended_price": None,
            "recommended_range": None,
            "max_expected_profit": None,
            "method": "failed",
            "details": segment_info,
            "logic_steps": [],
            "risk_text": "The available data did not support a stable and relevant historical price reference.",
        }
        rec["recommendation_text"] = build_recommendation_text(rec)
        return rec

    if segment_band is None and segment_band_type == "point":
        segment_band = narrow_band_around_point(segment_anchor, width_pct=0.05)

    if segment_band is None and segment_anchor is not None:
        rec = {
            "status": "ok",
            "segment_source": segment_source,
            "segment_stats": segment_stats,
            "segment_band": None,
            "segment_band_type": "point",
            "segment_anchor": segment_anchor,
            "profit_band_global": None,
            "intersection_band": None,
            "recommended_price": segment_anchor,
            "recommended_range": None,
            "max_expected_profit": None,
            "method": "point_anchor_fallback",
            "details": {
                **segment_info,
                "pred_metrics": pred_metrics,
                "global_median_price": global_median_price,
            },
        }
        rec["logic_steps"] = build_logic_steps(
            segment_source=segment_source,
            segment_band=None,
            segment_n=segment_stats["n"],
            segment_unique_n=segment_stats.get("unique_n"),
            global_profit_band=None,
            intersection_band=None,
            method=rec["method"],
            user_config=user_config,
            segment_band_type="point",
            segment_anchor=segment_anchor,
        )
        rec["risk_text"] = build_risk_text(
            segment_n=segment_stats["n"],
            segment_unique_n=segment_stats.get("unique_n"),
            pred_metrics=pred_metrics,
            segment_source=segment_source,
            global_median_price=global_median_price,
        )
        rec["recommendation_text"] = build_recommendation_text(rec)
        return rec

    if segment_band is None:
        rec = {
            "status": "no_reliable_segment_band",
            "segment_source": segment_source,
            "segment_stats": segment_stats,
            "segment_band": None,
            "segment_band_type": segment_band_type,
            "segment_anchor": segment_anchor,
            "profit_band_global": None,
            "intersection_band": None,
            "recommended_price": None,
            "recommended_range": None,
            "max_expected_profit": None,
            "method": "failed",
            "details": {
                **segment_info,
                "pred_metrics": pred_metrics,
                "global_median_price": global_median_price,
            },
            "logic_steps": [],
            "risk_text": "A relevant historical segment was found, but no usable price reference could be constructed from it.",
        }
        rec["recommendation_text"] = build_recommendation_text(rec)
        return rec

    global_profit = summarize_expected_profit(
        pred_dist=pred_dist,
        quantity=quantity,
        cost=cost,
        pct_of_max=global_profit_pct,
    )
    profit_band_global = global_profit["price_range_ge_pct_max"]
    intersection_band = intersect_ranges(segment_band, profit_band_global)

    if intersection_band is not None:
        constrained = constrained_profit_summary(
            pred_dist=pred_dist,
            quantity=quantity,
            cost=cost,
            allowed_range=intersection_band,
            pct_of_local_max=local_profit_pct,
        )

        recommended_price = constrained["optimal_price"]
        recommended_range = intersection_band
        method = "intersection_of_segment_and_global_profit_band"
        max_expected_profit = constrained["max_expected_profit"]
    else:
        constrained = constrained_profit_summary(
            pred_dist=pred_dist,
            quantity=quantity,
            cost=cost,
            allowed_range=segment_band,
            pct_of_local_max=local_profit_pct,
        )

        recommended_price = constrained["optimal_price"]
        recommended_range = constrained["price_range_ge_pct_local_max"]
        method = "local_profit_optimization_within_segment_band"
        max_expected_profit = constrained["max_expected_profit"]

    if recommended_price is None:
        low, high = segment_band
        recommended_price = float((low + high) / 2.0)
        if recommended_range is None:
            recommended_range = segment_band
        method = f"{method}_fallback_midpoint"

    if recommended_range is None:
        recommended_range = segment_band

    rec = {
        "status": "ok",
        "segment_source": segment_source,
        "segment_stats": segment_stats,
        "segment_band": segment_band,
        "segment_band_type": segment_band_type,
        "segment_anchor": segment_anchor,
        "profit_band_global": profit_band_global,
        "intersection_band": intersection_band,
        "recommended_price": recommended_price,
        "recommended_range": recommended_range,
        "max_expected_profit": max_expected_profit,
        "method": method,
        "details": {
            **segment_info,
            "global_profit_summary": global_profit,
            "constrained_summary": constrained,
            "pred_metrics": pred_metrics,
            "global_median_price": global_median_price,
        },
    }

    rec["logic_steps"] = build_logic_steps(
        segment_source=segment_source,
        segment_band=segment_band,
        segment_n=segment_stats["n"],
        segment_unique_n=segment_stats.get("unique_n"),
        global_profit_band=profit_band_global,
        intersection_band=intersection_band,
        method=rec["method"].replace("_fallback_midpoint", ""),
        user_config=user_config,
        segment_band_type=segment_band_type,
        segment_anchor=segment_anchor,
    )

    rec["risk_text"] = build_risk_text(
        segment_n=segment_stats["n"],
        segment_unique_n=segment_stats.get("unique_n"),
        pred_metrics=pred_metrics,
        segment_source=segment_source,
        global_median_price=global_median_price,
    )

    rec["recommendation_text"] = build_recommendation_text(rec)
    return rec