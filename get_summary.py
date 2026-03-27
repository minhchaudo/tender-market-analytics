import numpy as np
from db_cleaning_map import COUNTRY_TO_REGION

def to_scalar(x):
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return float(x)

def exp_safe(v):
    return float(np.exp(np.clip(v, -700, 700)))

def get_quantiles_df(df, target, key=None, value=None, value_ood="Other", quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    if value is None:
        quantiles = df[target].quantile(quantiles)
    else:
        if value == value_ood:
            return
        quantiles = df[df[key] == value][target].quantile(quantiles)
    return quantiles.to_dict()

def get_top_competitors(df, target, compete, n_top_competes=10):
    n_competes = len(df[compete].unique())
    value_competes = (df.groupby(compete, as_index=False)[target]
                      .sum()
                      .sort_values(target)
                      .head(n_top_competes)
                      .set_index(compete)
                      )
    return n_competes, value_competes[target].to_dict()    

def get_price_quantiles(pred_dist, quantile_levels):
    quantiles_vnd = {}
    for q in quantile_levels:
        log_val = to_scalar(pred_dist.ppf(q)) 
        val = exp_safe(log_val) * 1000  
        quantiles_vnd[q] = val

    mu_log = to_scalar(getattr(pred_dist, "mu", np.nan))
    sigma_log = to_scalar(getattr(pred_dist, "sigma", np.nan))

    if np.isfinite(mu_log) and np.isfinite(sigma_log):
        mean = np.exp(mu_log + 0.5 * sigma_log**2) * 1000
        std = np.sqrt((np.exp(sigma_log**2) - 1.0) * np.exp(2 * mu_log + sigma_log**2)) * 1000
    else:
        mean = np.nan
        std = np.nan

    return {
        "mean": float(mean),
        "std": float(std),
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
            "max_expected_profit": np.nan,
            "optimal_price": np.nan,
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


def get_info(df, query, user_config, pred_metrics, pred_dist, quantiles=[0.10, 0.25, 0.5, 0.75, 0.90], filtered=False):
    region_of_origin = COUNTRY_TO_REGION.get(user_config["country_of_origin"], "Other")
    quantiles_all = get_quantiles_df(df, "unit_price", quantiles=quantiles)
    quantiles_manufacturer = get_quantiles_df(df, "unit_price", "manufacturer", user_config["manufacturer"], quantiles=quantiles)
    quantiles_country_of_origin = get_quantiles_df(df, "unit_price", "country_of_origin", user_config["country_of_origin"], quantiles=quantiles)
    quantiles_region_of_origin = get_quantiles_df(df, "unit_price", "region_of_origin", region_of_origin, quantiles=quantiles)
    quantiles_investor = get_quantiles_df(df, "unit_price", "investor", user_config["investor"], quantiles=quantiles)
    n_competes, top_competes = get_top_competitors(df, "unit_price", "contractor_name", 10)

    msg = f"""For the product defined by query {query}{" and filters" if filtered else ""}, we retrieved {len(df)} rows of historical data, which represent winning bids of this product. Based on this data, we would like to recommend the best pricing strategy for bid {user_config}.
    
    The quantiles of unit price (winning bid price) for this product are: {quantiles_all}.

    The quantiles of unit price by each category are as follows:
    - For investor {user_config["investor"]}: {quantiles_investor}.
    - For manufacturer {user_config["manufacturer"]}: {quantiles_manufacturer}.
    - For country_of_origin {user_config["country_of_origin"]}: {quantiles_country_of_origin}.
    - For region_of_origin {COUNTRY_TO_REGION.get(user_config["country_of_origin"], "Other")}: {quantiles_region_of_origin}.

    Investor {user_config["investor"]} previously had {n_competes} contractors for this product. The top contractors are {list(top_competes.keys())}. The quantiles of unit price set by each top contractor for this product are as follows: 
    """
    for c in list(top_competes.keys()):
        msg += f"""
        - Contractor {c}: {get_quantiles_df(df, "unit_price", "contractor_name", c, quantiles=quantiles)}
        """

    pred_price = get_price_quantiles(pred_dist, quantiles)
    msg += f"""
    Using all historical data, we trained a ML model to predict winning unit price distribution for a new bid. Model performance is as follows: {pred_metrics}. For the given bid, the predicted distribution is as follows: mean {pred_price["mean"]}, std {pred_price["std"]}, quantiles {pred_price["quantiles"]}."""
    if "cost" in user_config and user_config["cost"] is not None:
        expected_profit = summarize_expected_profit(pred_dist, user_config["quantity"], user_config["cost"])
        msg += f"""We also calculate a proxy for the expected profit: (unit_price x quantity - cost) x (1 - q), where q is the quantile of a given unit price. Based on this, the optimal unit price is {expected_profit["optimal_price"]} with maximal expected profit {pred_price["max_expected_profit"]}. The unit price range so that the expected profit is >= 90% of the maximal expected profit is {expected_profit["price_range_ge_90pct"]}. The unit price range so that the expected profit is >= 50% of the maximal expected profit is {expected_profit["price_range_ge_50pct"]}.
        """
    return msg
