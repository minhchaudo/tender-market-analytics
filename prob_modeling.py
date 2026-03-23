import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skpro.regression.residual import ResidualDouble
from skpro.utils.numpy import flatten_to_1D_if_colvector
from skpro.utils.sklearn import prep_skl_df
from math import prod
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "investor",
    "log_quantity",
    "manufacturer",
    "origin",
    "province",
    "year",
    "month",
]

CAT_FEATURES = ["investor", "manufacturer", "origin", "province"]
NUM_FEATURES = ["log_quantity", "year", "month"]
TARGET = "unit_price"
DATE_COL = "date"

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500, 700],
    "max_depth": [None, 5, 8, 12, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, 0.8],
    "random_state": [0],
    "n_jobs": [-1],
}

N_SEARCH = 30
RANDOM_STATE = 0
random.seed(RANDOM_STATE)

class MyResidualDouble(ResidualDouble):
    def __init__(self,
        estimator,
        estimator_resid=None,
        residual_trafo="absolute",
        distr_type="Normal",
        distr_loc_scale_name=None,
        distr_params=None,
        use_y_pred=False,
        cv=None,
        min_scale=1e-10):
        super().__init__(estimator, estimator_resid, residual_trafo, distr_type, distr_loc_scale_name, distr_params, use_y_pred, cv, min_scale)
    
    def _fit(self, X, y):
        est = self.estimator_
        est_r = self.estimator_resid_
        residual_trafo = self.residual_trafo
        cv = self.cv
        use_y_pred = self.use_y_pred

        self._y_cols = ["y_pred"] # for HM mean + RF residual when use_y_pred=True

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # flatten column vector to 1D array to avoid sklearn complaints
        y = y.values
        y = flatten_to_1D_if_colvector(y)

        est.fit(X, y)

        if cv is None:
            y_pred = est.predict(X)
        else:
            y_pred = self._predict_residuals_cv(X, y, cv, est)

        if residual_trafo == "absolute":
            resids = np.abs(y - y_pred)
        elif residual_trafo == "squared":
            resids = (y - y_pred) ** 2
        else:
            resids = residual_trafo(y - y_pred)

        resids = flatten_to_1D_if_colvector(resids)

        if use_y_pred:
            y_ix = {"index": X.index, "columns": self._y_cols}
            X_r = pd.concat([X, pd.DataFrame(y_pred, **y_ix)], axis=1)
        else:
            X_r = X

        # coerce X to pandas DataFrame with string column names
        X_r = prep_skl_df(X_r, copy_df=True)

        est_r.fit(X_r, resids)

        return self


class HierMean(BaseEstimator):
    def __init__(self, **args):
        super().__init__()
        pass
    
    def fit(self, train_df, y):
        self.global_mean = train_df[TARGET].mean()

        self.mean_mcpi = train_df.groupby(
            ["manufacturer", "origin", "province", "investor"]
        )[TARGET].mean()

        self.mean_mcp = train_df.groupby(
            ["manufacturer", "origin", "province"]
        )[TARGET].mean()

        self.mean_mc = train_df.groupby(
            ["manufacturer", "origin"]
        )[TARGET].mean()

        self.mean_m = train_df.groupby(
            ["manufacturer"]
        )[TARGET].mean()

        self.mean_c = train_df.groupby(
            ["origin"]
        )[TARGET].mean()

        self.mean_i = train_df.groupby(
            ["investor"]
        )[TARGET].mean()

        self.mean_p = train_df.groupby(
            ["province"]
        )[TARGET].mean()

        return self 

    def predict(self, test_df):
        pred = pd.Series(
            test_df.set_index(
                ["manufacturer", "origin", "province", "investor"]
            ).index.map(self.mean_mcpi),
            index=test_df.index,
            dtype=float
        )

        missing = pred.isna()
        pred.loc[missing] = pd.Series(
            test_df.loc[missing].set_index(
                ["manufacturer", "origin", "province"]
            ).index.map(self.mean_mcp),
            index=test_df.loc[missing].index,
            dtype=float
        )

        missing = pred.isna()
        pred.loc[missing] = pd.Series(
            test_df.loc[missing].set_index(
                ["manufacturer", "origin"]
            ).index.map(self.mean_mc),
            index=test_df.loc[missing].index,
            dtype=float
        )

        missing = pred.isna()
        pred.loc[missing] = test_df.loc[missing, "manufacturer"].map(self.mean_m)

        missing = pred.isna()
        pred.loc[missing] = test_df.loc[missing, "origin"].map(self.mean_c)

        missing = pred.isna()
        pred.loc[missing] = test_df.loc[missing, "investor"].map(self.mean_i)

        missing = pred.isna()
        pred.loc[missing] = test_df.loc[missing, "province"].map(self.mean_p)

        pred = pred.fillna(self.global_mean)

        return pred.to_numpy()
    
def get_train_val_test_split(df):
    df_model = df[FEATURES + [TARGET, DATE_COL]].copy()
    df_model = df_model.dropna(subset=[TARGET, DATE_COL] + FEATURES)
    df_model = df_model.sort_values(DATE_COL).reset_index(drop=True)

    n = len(df_model)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_model.iloc[:train_end].copy()
    val_df   = df_model.iloc[train_end:val_end].copy()
    test_df  = df_model.iloc[val_end:].copy()
    return train_df, val_df, test_df


def get_pipeline_wrap(preprocessor, model_class):
    def get_pipeline(**args):
        return Pipeline([
        ("prep", preprocessor),
        ("model", model_class(**args))
    ])
    return get_pipeline


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def decode_product_index(idx, pools):
    sizes = [len(pool) for pool in pools]
    values = [None] * len(pools)

    for i in range(len(pools) - 1, -1, -1):
        idx, rem = divmod(idx, sizes[i])
        values[i] = pools[i][rem]

    return tuple(values)


def sample_paired_param_combinations(mean_param_grid, residual_param_grid, n_iter, random_state=None):
    rng = np.random.default_rng(random_state)

    mean_keys = list(mean_param_grid.keys())
    residual_keys = list(residual_param_grid.keys())

    mean_values = [mean_param_grid[k] for k in mean_keys]
    residual_values = [residual_param_grid[k] for k in residual_keys]
    pools = mean_values + residual_values

    sizes = [len(pool) for pool in pools]
    total = prod(sizes)

    n_samples = min(n_iter, total)
    chosen_idx = rng.choice(total, size=n_samples, replace=False)

    mean_params = []
    residual_params = []

    split = len(mean_keys)

    for idx in chosen_idx:
        combo = decode_product_index(int(idx), pools)

        mean_params.append({
            k: v for k, v in zip(mean_keys, combo[:split])
        })
        residual_params.append({
            k: v for k, v in zip(residual_keys, combo[split:])
        })

    return mean_params, residual_params

def train_scale_model(
    train_df, val_df, test_df,
    features, target,
    model_name,
    mean_model_class,
    residual_model_class,
    mean_param_list,
    residual_param_list,
    use_y_pred
):  
    
    X_train, y_train = train_df[features], train_df[[target]]
    X_val, y_val = val_df[features], val_df[[target]]

    best_val_nll = np.inf
    best_mean_params = None
    best_residual_params = None

    for mean_params, residual_params in tqdm(
        zip(mean_param_list, residual_param_list),
        total=len(mean_param_list),
        desc=model_name,
    ):

        reg = MyResidualDouble(
            estimator=mean_model_class(**mean_params),
            estimator_resid=residual_model_class(**residual_params),        use_y_pred=use_y_pred,
            cv=KFold(3)
        )

        reg._fit(X_train, y_train)

        val_dist = reg._predict_proba(X_val)
        val_nll = -np.mean(val_dist.log_pdf(y_val))
        

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_mean_params = mean_params
            best_residual_params = residual_params

    trainval_df = pd.concat([train_df, val_df], axis=0)
    X_trainval, y_trainval = trainval_df[features], trainval_df[[target]]
    X_test, y_test = test_df[features], test_df[[target]]

    best_reg = MyResidualDouble(
        estimator=mean_model_class(**best_mean_params),
        estimator_resid=residual_model_class(**best_residual_params),        use_y_pred=use_y_pred,
        cv=KFold(3),
    )
    best_reg._fit(X_trainval, y_trainval)

    test_dist = best_reg._predict_proba(X_test)
    test_pred = test_dist.mu
    test_nll = -np.mean(test_dist.log_pdf(y_test))

    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    X_all, y_all = all_df[features], all_df[[target]]

    final_reg = MyResidualDouble(
        estimator=mean_model_class(**best_mean_params),
        estimator_resid=residual_model_class(**best_residual_params),
        use_y_pred=use_y_pred,
        cv=KFold(3),
    )
    final_reg._fit(X_all, y_all)

    return {
        "model": model_name,
        "best_mean_params": best_mean_params,
        "best_residual_params": best_residual_params,
        "test_nll": test_nll,
        "test_mean_pred": test_pred,
        "reg": final_reg,
    }


def predict_winning_price_scale_model(
    train_df, val_df, test_df,
    features, target,
    mean_model_class,
    residual_model_class,
    mean_param_grid,
    residual_param_grid,
    model_name,
    use_y_pred=False
):
    
    mean_params, residual_params = sample_paired_param_combinations(
        mean_param_grid, residual_param_grid, N_SEARCH, RANDOM_STATE
    )

    res = train_scale_model(
        train_df, val_df, test_df,
        features, target,
        model_name=model_name,
        mean_model_class=mean_model_class,
        residual_model_class=residual_model_class,
        mean_param_list=mean_params,
        residual_param_list=residual_params,
        use_y_pred=use_y_pred
    )

    test_mu = res["test_mean_pred"]

    reg = regression_metrics(test_df[[target]], test_mu)

    test_summary = {
        "MAE": reg["MAE"],
        "RMSE": reg["RMSE"],
        "R2": reg["R2"],
        "NLL": res["test_nll"],
    }

    return res["reg"], test_summary

def prep_df(df):
    df = df.copy()
    df["log_quantity"] = np.log1p(df["quantity"])
    df["date"] = pd.to_datetime(df["posting_date"], dayfirst=True)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["unit_price"] = df["unit_price"] / 1000
    return df

def train_HM_RF(df, need_prep=True):
    if need_prep:
        df = prep_df(df)
    train_df, val_df, test_df = get_train_val_test_split(df)
    # target used by HierMean in training, not used by RF 
    features = FEATURES + [TARGET] 
    rf_prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES + ["y_pred"]), # use_y_pred=True
        ]
    )
    residual_model_class = get_pipeline_wrap(rf_prep, RandomForestRegressor)
    model, metrics = predict_winning_price_scale_model(train_df, val_df, test_df, features, TARGET, HierMean, residual_model_class, {}, RF_PARAM_GRID, "HM_RF", use_y_pred=True)
    return model, metrics

def predict_HM_RF(X_hat, model):
    pred_dist = model._predict_proba(X_hat)
    return pred_dist

if __name__ == "__main__":
    df = pd.read_csv("modeling_test_data/hiv_scrape.csv")
    model, metrics = train_HM_RF(df, need_prep=False)
    print(metrics)


