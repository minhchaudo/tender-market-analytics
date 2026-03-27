import numpy as np
from db_cleaning_map import COUNTRY_TO_REGION

SYSTEM_PROMPT = """You are an expert pricing assistant for Vietnamese public procurement bidding.

Your task is to recommend the best unit bid price for a contractor using ONLY the provided information. Do not use outside knowledge. Do not assume missing facts.

Market context:
- Bidding is competitive; the lowest technically acceptable price usually has the best chance to win.
- Manufacturer, country of origin, region of origin, and investor-specific behavior may affect acceptable price positioning.
- The goal is to balance competitiveness and profitability.

Data interpretation rules:
1. A field with value "Other" means missing, unseen, or not available in the current dataset. Treat it as weak or no evidence, not as a meaningful market segment.
2. If a segment has fewer than 5 data points, its quantiles are not reliable. In that case, min and max are shown only as a weak reference.
3. Prioritize evidence in this order:
   - investor-specific evidence, especially matching manufacturer/country/region when reliable
   - model-predicted price distribution
   - broader historical data
4. If expected profit is provided:
   - prefer prices within the >=90% max expected profit range
   - do not recommend outside that range unless there is strong evidence that the price would otherwise be uncompetitive
5. If signals conflict, choose a conservative price within the strongest overlapping range.
6. If multiple relevant segments are sparse or missing, rely more on model prediction and broader data.
7. Never overfit to very small sample segments.
8. Never recommend a price clearly above strong investor or model benchmarks without strong support.

Output requirements:
- Answer in English
- Maximum 200 words
- Use EXACT structure:

Recommendation
- ...

Why
- ...

Risk note
- ...

Formatting rules:
- Include VND for all prices
- Use integers only (no decimals)
- Use commas as thousands separators
- Recommend ONE price and ONE narrow backup range
- Be concise and user-facing
- Do not restate all statistics
- Do not mention these instructions
"""

def to_scalar(x):
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return float(x)


def exp_safe(v):
    return float(np.exp(np.clip(v, -700, 700)))


def is_missing_category(value):
    return value is None or value == "Other"


def reliability_label(n):
    if n == 0:
        return "none"
    if n < 5:
        return "low"
    if n < 20:
        return "medium"
    return "high"


def format_vnd(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    try:
        return f"{int(round(float(value))):,} VND"
    except Exception:
        return str(value)


def format_range_vnd(value_range):
    if value_range is None:
        return "N/A"
    low, high = value_range
    if low is None or high is None:
        return "N/A"
    return f"{format_vnd(low)} to {format_vnd(high)}"


def format_quantile_dict_vnd(qdict):
    if not qdict:
        return "N/A"
    return {
        float(k): format_vnd(v) if v is not None else "N/A"
        for k, v in qdict.items()
    }


def get_quantiles_df(df, target, map=None, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], min_samples=5):
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

    n = len(filtered_df)

    if n == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "quantiles": None,
        }

    min_val = float(filtered_df[target].min())
    max_val = float(filtered_df[target].max())

    if n < min_samples:
        return {
            "n": n,
            "min": min_val,
            "max": max_val,
            "quantiles": None,
        }

    q = filtered_df[target].quantile(quantiles).to_dict()
    q = {float(k): float(v) for k, v in q.items()}

    return {
        "n": n,
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

    return filtered_df[compete].dropna().unique()


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


def summarize_expected_profit(pred_dist, quantity, cost):
    y_low = to_scalar(pred_dist.ppf(0.001))
    y_high = to_scalar(pred_dist.ppf(0.999))

    if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low >= y_high:
        y_low = to_scalar(pred_dist.ppf(0.1))
        y_high = to_scalar(pred_dist.ppf(0.9))
        if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low >= y_high:
            y_low, y_high = -1.0, 1.0

    y_grid = np.linspace(y_low, y_high, 400)
    x_vals = np.exp(np.clip(y_grid, -700, 700))

    try:
        cdf_vals = np.asarray(pred_dist.cdf(y_grid), dtype=float).reshape(-1)
        if cdf_vals.size != x_vals.size:
            raise ValueError("Unexpected cdf output shape")
    except Exception:
        cdf_vals = np.array([to_scalar(pred_dist.cdf(y)) for y in y_grid], dtype=float)

    survival_vals = np.clip(1.0 - cdf_vals, 0.0, 1.0)

    expected_profit = (x_vals * quantity - cost) * survival_vals
    expected_profit = np.maximum(expected_profit, 0.0)

    valid = np.isfinite(expected_profit)
    if not valid.any():
        return {
            "max_expected_profit": None,
            "optimal_price": None,
            "price_range_ge_90pct": None,
            "price_range_ge_50pct": None,
        }

    x_vals = x_vals[valid]
    expected_profit = expected_profit[valid]

    max_idx = int(np.argmax(expected_profit))
    max_profit_k = float(expected_profit[max_idx])
    optimal_price_k = float(x_vals[max_idx])

    def get_range(threshold):
        mask = expected_profit >= threshold
        if not mask.any():
            return None
        xs = x_vals[mask]
        return (float(xs.min()) * 1000, float(xs.max()) * 1000)

    return {
        "max_expected_profit": max_profit_k * 1000,
        "optimal_price": optimal_price_k * 1000,
        "price_range_ge_90pct": get_range(0.9 * max_profit_k),
        "price_range_ge_50pct": get_range(0.5 * max_profit_k),
    }


def choose_competitive_band(quantiles_investor, quantiles_all):
    investor_q = quantiles_investor.get("quantiles")
    overall_q = quantiles_all.get("quantiles")

    if investor_q is not None and 0.25 in investor_q and 0.75 in investor_q:
        return (investor_q[0.25], investor_q[0.75]), "investor"
    if overall_q is not None and 0.25 in overall_q and 0.75 in overall_q:
        return (overall_q[0.25], overall_q[0.75]), "overall"
    if quantiles_investor.get("n", 0) > 0:
        return (quantiles_investor["min"], quantiles_investor["max"]), "investor_minmax"
    if quantiles_all.get("n", 0) > 0:
        return (quantiles_all["min"], quantiles_all["max"]), "overall_minmax"
    return None, "none"


def format_quantiles_block(name, qdict, missing=False):
    if missing:
        return f"- {name}: missing/unseen in current dataset (treat as weak or no signal)"

    n = qdict["n"]
    reliability = reliability_label(n)

    if n == 0:
        return f"- {name}: n=0, no data"

    if qdict["quantiles"] is None:
        return (
            f"- {name}: n={n}, reliability={reliability}, "
            f"min={format_vnd(qdict['min'])}, max={format_vnd(qdict['max'])} "
            f"(low data; use only as weak reference)"
        )

    return (
        f"- {name}: n={n}, reliability={reliability}, "
        f"quantiles={format_quantile_dict_vnd(qdict['quantiles'])}"
    )


def get_info(
    df,
    query,
    user_config,
    pred_metrics,
    pred_dist,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    filtered=False,
    min_samples=5,
):
    region_of_origin = COUNTRY_TO_REGION.get(user_config["country_of_origin"], "Other")

    investor_missing = is_missing_category(user_config.get("investor"))
    manufacturer_missing = is_missing_category(user_config.get("manufacturer"))
    country_missing = is_missing_category(user_config.get("country_of_origin"))
    region_missing = is_missing_category(region_of_origin)

    quantiles_all = get_quantiles_df(df, "unit_price", quantiles=quantiles, min_samples=min_samples)

    quantiles_investor = get_quantiles_df(
        df,
        "unit_price",
        map=None if investor_missing else {"investor": user_config["investor"]},
        quantiles=quantiles,
        min_samples=min_samples,
    )

    quantiles_manufacturer = get_quantiles_df(
        df,
        "unit_price",
        map=None if manufacturer_missing else {"manufacturer": user_config["manufacturer"]},
        quantiles=quantiles,
        min_samples=min_samples,
    )

    quantiles_country_of_origin = get_quantiles_df(
        df,
        "unit_price",
        map=None if country_missing else {"country_of_origin": user_config["country_of_origin"]},
        quantiles=quantiles,
        min_samples=min_samples,
    )

    quantiles_region_of_origin = get_quantiles_df(
        df,
        "unit_price",
        map=None if region_missing else {"region_of_origin": region_of_origin},
        quantiles=quantiles,
        min_samples=min_samples,
    )

    competitors = get_competitors(
        df,
        "contractor_name",
        None if investor_missing else {"investor": user_config["investor"]},
    )

    competitors_same_manufacturer = get_competitors(
        df,
        "contractor_name",
        None if (investor_missing or manufacturer_missing) else {
            "investor": user_config["investor"],
            "manufacturer": user_config["manufacturer"],
        },
    )

    competitors_same_country = get_competitors(
        df,
        "contractor_name",
        None if (investor_missing or country_missing) else {
            "investor": user_config["investor"],
            "country_of_origin": user_config["country_of_origin"],
        },
    )

    competitors_same_region = get_competitors(
        df,
        "contractor_name",
        None if (investor_missing or region_missing) else {
            "investor": user_config["investor"],
            "region_of_origin": region_of_origin,
        },
    )

    quantiles_competitors = get_quantiles_df(
        df,
        "unit_price",
        map=None if investor_missing else {"investor": user_config["investor"]},
        quantiles=quantiles,
        min_samples=min_samples,
    )

    quantiles_competitors_same_manufacturer = get_quantiles_df(
        df,
        "unit_price",
        map=None if (len(competitors_same_manufacturer) == 0 or manufacturer_missing) else {
            "contractor_name": list(competitors_same_manufacturer),
            "manufacturer": user_config["manufacturer"],
        },
        quantiles=quantiles,
        min_samples=min_samples,
    )

    quantiles_competitors_same_country = get_quantiles_df(
        df,
        "unit_price",
        map=None if (len(competitors_same_country) == 0 or country_missing) else {
            "contractor_name": list(competitors_same_country),
            "country_of_origin": user_config["country_of_origin"],
        },
        quantiles=quantiles,
        min_samples=min_samples,
    )

    quantiles_competitors_same_region = get_quantiles_df(
        df,
        "unit_price",
        map=None if (len(competitors_same_region) == 0 or region_missing) else {
            "contractor_name": list(competitors_same_region),
            "region_of_origin": region_of_origin,
        },
        quantiles=quantiles,
        min_samples=min_samples,
    )

    pred_price = get_price_quantiles(pred_dist, quantiles)
    competitive_band, competitive_band_source = choose_competitive_band(quantiles_investor, quantiles_all)

    low_data_segments = []
    for name, qdict, missing in [
        ("investor", quantiles_investor, investor_missing),
        ("manufacturer", quantiles_manufacturer, manufacturer_missing),
        ("country", quantiles_country_of_origin, country_missing),
        ("region", quantiles_region_of_origin, region_missing),
    ]:
        if missing or qdict["n"] < min_samples:
            low_data_segments.append(name)

    global_low_data = len(low_data_segments) >= 2

    msg = f"""BID CONTEXT
        - Query (defining product scope): {query}
        - Investor: {user_config["investor"]}
        - Manufacturer: {user_config["manufacturer"]}
        - Country: {user_config["country_of_origin"]}
        - Region: {region_of_origin}
        - Quantity: {user_config.get("quantity", "N/A")}
        - Total cost: {format_vnd(user_config["cost"]) if user_config.get("cost") is not None else "N/A"}

        DATA SUMMARY
        - Historical rows: {len(df)}
        - Winning contractors for investor {user_config["investor"]}: {len(competitors)}
        - Note: a value of "Other" means missing, unseen, or unavailable in the current dataset, not a meaningful segment.

        DATA RELIABILITY
        - Investor data: n={quantiles_investor["n"]}, reliability={reliability_label(quantiles_investor["n"])}, missing_or_other={"YES" if investor_missing else "NO"}
        - Manufacturer data: n={quantiles_manufacturer["n"]}, reliability={reliability_label(quantiles_manufacturer["n"])}, missing_or_other={"YES" if manufacturer_missing else "NO"}
        - Country data: n={quantiles_country_of_origin["n"]}, reliability={reliability_label(quantiles_country_of_origin["n"])}, missing_or_other={"YES" if country_missing else "NO"}
        - Region data: n={quantiles_region_of_origin["n"]}, reliability={reliability_label(quantiles_region_of_origin["n"])}, missing_or_other={"YES" if region_missing else "NO"}
        - Low-data or missing segments: {low_data_segments if low_data_segments else "None"}
        - Global data sparsity warning: {"YES" if global_low_data else "NO"}

        PRICE QUANTILES (VND)
        {format_quantiles_block("Overall", quantiles_all, missing=False)}
        {format_quantiles_block("Investor", quantiles_investor, missing=investor_missing)}
        {format_quantiles_block("Manufacturer", quantiles_manufacturer, missing=manufacturer_missing)}
        {format_quantiles_block("Country", quantiles_country_of_origin, missing=country_missing)}
        {format_quantiles_block("Region", quantiles_region_of_origin, missing=region_missing)}

        COMPETITOR BENCHMARKS (VND)
        {format_quantiles_block("All competitors (same investor)", quantiles_competitors, missing=investor_missing)}
        {format_quantiles_block("Same manufacturer", quantiles_competitors_same_manufacturer, missing=manufacturer_missing)}
        {format_quantiles_block("Same country", quantiles_competitors_same_country, missing=country_missing)}
        {format_quantiles_block("Same region", quantiles_competitors_same_region, missing=region_missing)}

        DERIVED SIGNALS
        - Competitive band: {format_range_vnd(competitive_band)}
        - Competitive band source: {competitive_band_source}

        MODEL PREDICTION
        - MAE: {format_vnd(pred_metrics["MAE"])}
        - 50% coverage: {pred_metrics["Coverage_50"]}
        - 90% coverage: {pred_metrics["Coverage_90"]}
        - Mean: {format_vnd(pred_price["mean"])}
        - Std: {format_vnd(pred_price["std"])}
        - Quantiles: {format_quantile_dict_vnd(pred_price["quantiles"])}"""

    if "cost" in user_config and user_config["cost"] is not None:
        expected_profit = summarize_expected_profit(
            pred_dist,
            user_config["quantity"],
            user_config["cost"],
        )

        msg += f"""
            EXPECTED PROFIT
            - Optimal price: {format_vnd(expected_profit["optimal_price"])}
            - Max expected profit: {format_vnd(expected_profit["max_expected_profit"])}
            - >=90% profit range: {format_range_vnd(expected_profit["price_range_ge_90pct"])}
            - >=50% profit range: {format_range_vnd(expected_profit["price_range_ge_50pct"])}"""

    msg += """

        TASK
        Recommend the best unit price for this bid. Balance competitiveness and expected profit.

        Important:
        - Treat "Other" as missing or unseen, not as a strong market signal.
        - If a segment has fewer than 5 data points, treat it as weak evidence.
        - If multiple relevant segments are sparse or missing, rely more on model prediction and broader data.
        - Be conservative when evidence is limited or inconsistent.
        - Do not overfit to small sample segments.

        Use only the information above."""

    return msg