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
from sklearn.model_selection import KFold
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import clone
import copy
from sklearn.model_selection import ParameterSampler
from itertools import combinations
import warnings

warnings.filterwarnings(
    "ignore",
    message="The total space of parameters .* is smaller than n_iter"
)

FEATURES = [
    "investor",
    "log_quantity",
    "manufacturer",
    "country_of_origin",
    "province",
    "year",
    "month",
]

CAT_FEATURES = ["investor", "manufacturer", "country_of_origin", "province"]
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

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500, 700],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 3, 5],
    "random_state": [0],
    "n_jobs": [-1],
}

LR_PARAM_GRID = {
    "fit_intercept": [True],
}

N_SEARCH = 30
RANDOM_STATE = 0
random.seed(RANDOM_STATE)

class HierMeanFeatureSet:
    def __init__(
        self,
        target_col="unit_price",
        fields=None,
        weights=None,
    ):
        self.target_col = target_col
        self.fields = fields or [
            "manufacturer",
            "country_of_origin",
            # "province",
            # "investor",
        ]
        self.weights = weights or {
            "manufacturer": 3.0,
            "country_of_origin": 2.0,
            "investor": 1.0,
            "province": 0.5,
        }

    def _all_subsets(self):
        for r in range(1, len(self.fields) + 1):
            yield from combinations(self.fields, r)

    def _subset_weight(self, subset):
        return float(sum(self.weights.get(col, 0.0) for col in subset))

    def get_feature_names_out(self):
        return (
            ["hier_mean", "hier_count"]
            + [f"{f}_used" for f in self.fields]
        )

    def fit(self, train_df):
        y = self.target_col

        self.global_mean_ = float(train_df[y].mean())
        self.global_count_ = int(len(train_df))
        self.subsets_ = list(self._all_subsets())
        self.stats_ = {}

        for subset in self.subsets_:
            cols = list(subset)
            grp = (
                train_df.groupby(cols, dropna=False)[y]
                .agg(["sum", "count"])
                .rename(columns={"sum": "sum_", "count": "count_"})
            )
            grp["mean_"] = grp["sum_"] / grp["count_"]
            self.stats_[subset] = grp

        return self

    def _select_best(self, candidates, index, is_train):
        pred = pd.Series(
            self.global_mean_,
            index=index,
            dtype=float,
        )
        pred_count = pd.Series(
            self.global_count_ - 1 if is_train else self.global_count_,
            index=index,
            dtype=float,
        )

        used = {
            f: pd.Series(0, index=index, dtype=int)
            for f in self.fields
        }

        if not candidates:
            return pred, pred_count, used

        candidates = pd.concat(candidates, axis=0, ignore_index=True)

        candidates = candidates.sort_values(
            by=["_row_idx", "_subset_weight", "count"],
            ascending=[True, False, False],
            kind="mergesort",
        )

        best = candidates.drop_duplicates(subset="_row_idx", keep="first")

        pred.loc[best["_row_idx"]] = best["mean"].to_numpy()
        pred_count.loc[best["_row_idx"]] = best["count"].to_numpy()

        for i, subset in zip(best["_row_idx"], best["_subset"]):
            for f in subset:
                used[f].loc[i] = 1

        return pred, pred_count, used

    def transform_train(self, train_df):
        y = self.target_col
        idx = train_df.index
        candidates = []

        for subset in self.subsets_:
            cols = list(subset)
            tab = self.stats_[subset][["sum_", "count_"]].reset_index()

            merged = train_df[cols + [y]].merge(
                tab, on=cols, how="left", sort=False
            )

            full_n = merged["count_"].astype(float)
            loo_n = full_n - 1

            loo_mean = (
                (merged["sum_"].astype(float) - merged[y].astype(float))
                / loo_n
            ).where(loo_n > 0)

            cand = pd.DataFrame({
                "_row_idx": idx,
                "mean": loo_mean,
                "count": loo_n.where(loo_n > 0, 0),
                "_subset_weight": self._subset_weight(subset),
                "_subset": [subset] * len(idx),
            })

            cand = cand[cand["mean"].notna()]
            if not cand.empty:
                candidates.append(cand)

        pred, pred_count, used = self._select_best(
            candidates, idx, is_train=True
        )

        out = train_df.copy()
        out["hier_mean"] = pred
        out["hier_count"] = pred_count

        for f in self.fields:
            out[f"{f}_used"] = used[f]

        return out

    def transform_test(self, test_df):
        idx = test_df.index
        candidates = []

        for subset in self.subsets_:
            cols = list(subset)
            tab = self.stats_[subset][["mean_", "count_"]].reset_index()

            merged = test_df[cols].merge(
                tab, on=cols, how="left", sort=False
            )

            cand = pd.DataFrame({
                "_row_idx": idx,
                "mean": pd.to_numeric(merged["mean_"], errors="coerce"),
                "count": pd.to_numeric(merged["count_"], errors="coerce"),
                "_subset_weight": self._subset_weight(subset),
                "_subset": [subset] * len(idx),
            })

            cand = cand[cand["mean"].notna()]
            if not cand.empty:
                candidates.append(cand)

        pred, pred_count, used = self._select_best(
            candidates, idx, is_train=False
        )

        out = test_df.copy()
        out["hier_mean"] = pred
        out["hier_count"] = pred_count

        for f in self.fields:
            out[f"{f}_used"] = used[f]

        return out
    
class ResidualDoubleCVSafe(ResidualDouble):
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
        super().__init__(estimator,
                        estimator_resid,
                        residual_trafo,
                        distr_type,
                        distr_loc_scale_name,
                        distr_params,
                        use_y_pred,
                        cv,
                        min_scale)

    def _predict_residuals_cv(self, X, y, cv, est):
        method = "predict"
        y_pred = y.copy()

        for X_train, y_train, X_test, y_test, tt_idx in cv(X, y):
            fitted_est = clone(est).fit(X_train, y_train)
            y_pred[tt_idx] = getattr(fitted_est, method)(X_test)
        return y_pred
    
    def _fit(self, X, y):
        est = self.estimator_
        est_r = self.estimator_resid_
        residual_trafo = self.residual_trafo
        cv = self.cv
        use_y_pred = self.use_y_pred

        self._y_cols = ["y_pred"]

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
    def __init__(self, fields=None, weights=None):
        super().__init__()
        self.fields = fields or [
            "manufacturer",
            "country_of_origin",
            # "province",
            # "investor",
        ]
        self.weights = weights or {
            "manufacturer": 3.0,
            "country_of_origin": 2.0,
            "investor": 1.0,
            "province": 0.5,
        }

    def _all_subsets(self):
        for r in range(1, len(self.fields) + 1):
            yield from combinations(self.fields, r)

    def _subset_weight(self, subset):
        return float(sum(self.weights.get(col, 0.0) for col in subset))

    def fit(self, X, y=None):
        target = y if y is not None else X[TARGET]

        df = X[self.fields].copy()
        df["_target_"] = target

        self.global_mean_ = float(df["_target_"].mean())
        self.tables_ = {}

        for subset in self._all_subsets():
            cols = list(subset)
            agg = (
                df.groupby(cols, dropna=False, observed=True)["_target_"]
                .agg(mean="mean", count="size")
                .reset_index()
            )
            agg["mean"] = agg["mean"].astype(float)
            agg["count"] = agg["count"].astype(int)
            agg["_subset_weight"] = self._subset_weight(subset)
            self.tables_[subset] = agg

        self.subsets_ = list(self.tables_.keys())
        return self

    def predict(self, X):
        Xf = X[self.fields].copy()
        pred = pd.Series(index=X.index, dtype=float)

        candidate_frames = []

        for subset in self.subsets_:
            cols = list(subset)

            matched = Xf[cols].merge(
                self.tables_[subset],
                on=cols,
                how="left",
                sort=False,
            )
            matched.index = Xf.index

            matched = matched[["mean", "count", "_subset_weight"]]
            matched = matched[matched["mean"].notna()].copy()
            if matched.empty:
                continue

            matched["_row_idx"] = matched.index
            candidate_frames.append(matched)

        if candidate_frames:
            candidates = pd.concat(candidate_frames, axis=0, ignore_index=True)

            candidates = candidates.sort_values(
                by=["_row_idx", "_subset_weight", "count"],
                ascending=[True, False, False],
                kind="mergesort",
            )

            best = candidates.drop_duplicates(subset="_row_idx", keep="first")
            pred.loc[best["_row_idx"]] = best["mean"].to_numpy()

        return pred.fillna(self.global_mean_).to_numpy()


def get_cv_fn(target, features, n_splits=3):
    def hiermean_cv(X, y):
        kf = KFold(n_splits)

        X = X.copy()
        y = pd.Series(y, index=X.index)

        for train_idx, test_idx in kf.split(X):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # build train df with target
            train_df = X_train.copy()
            train_df[target] = y_train

            # clone transformer (important!)
            hmt = HierMeanFeatureSet()
            hmt.fit(train_df)

            # transform
            train_tr = hmt.transform_train(train_df)
            test_tr = hmt.transform_test(X_test)

            # drop target from X
            X_train_tr = train_tr[features]
            X_test_tr = test_tr[features]

            yield X_train_tr, y_train, X_test_tr, y_test, test_idx
    return hiermean_cv

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

def train_mean_model(
    train_df, val_df, test_df,
    features, target,
    model_name,
    mean_model_class,
    mean_param_grid,
):  
    param_sampler = ParameterSampler(
        mean_param_grid,
        n_iter=N_SEARCH,
        random_state=RANDOM_STATE
    )

    hier_mean_transformer = HierMeanFeatureSet()
    hier_mean_transformer.fit(train_df)
    train_df_tr = hier_mean_transformer.transform_train(train_df)
    val_df_tr = hier_mean_transformer.transform_test(val_df)
    
    X_train, y_train = train_df_tr[features], train_df_tr[target]
    X_val, y_val = val_df_tr[features], val_df_tr[target]

    best_val_rmse = np.inf
    best_mean_params = None

    for mean_params in tqdm(list(param_sampler)):

        reg = mean_model_class(**mean_params)

        reg.fit(X_train, y_train)

        val_pred = reg.predict(X_val)
        val_metrics = regression_metrics(y_val, val_pred)
        val_rmse = val_metrics["RMSE"]
        

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_mean_params = mean_params

    trainval_df = pd.concat([train_df, val_df], axis=0)

    hier_mean_transformer = HierMeanFeatureSet()
    hier_mean_transformer.fit(trainval_df)
    trainval_df_tr = hier_mean_transformer.transform_train(trainval_df)
    test_df_tr = hier_mean_transformer.transform_test(test_df)
    X_trainval, y_trainval = trainval_df_tr[features], trainval_df_tr[target]
    X_test, y_test = test_df_tr[features], test_df_tr[target]

    best_reg = mean_model_class(**best_mean_params)
    best_reg.fit(X_trainval, y_trainval)

    test_pred = best_reg.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)

    return {
        "name": model_name,
        "class": mean_model_class,
        "best_mean_params": best_mean_params,
        "metrics": test_metrics
    }

def prep_df(df, predict=False):
    df = df.copy()
    df["log_quantity"] = np.log1p(df["quantity"])
    df["date"] = pd.to_datetime(df["closing_date"], dayfirst=True)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    if not predict:
        df["unit_price"] = df["unit_price"] / 1000
    return df


def train_scale_model(
    train_df, val_df, test_df,
    features, target,
    model_name,
    mean_model_class,
    residual_model_class,
    mean_params,
    residual_param_grid,
    use_y_pred
):  
    param_sampler = ParameterSampler(
        residual_param_grid,
        n_iter=N_SEARCH,
        random_state=RANDOM_STATE
    )

    hier_mean_transformer = HierMeanFeatureSet()
    hier_mean_transformer.fit(train_df)
    train_df_tr = hier_mean_transformer.transform_train(train_df)
    val_df_tr = hier_mean_transformer.transform_test(val_df)
    
    X_train, y_train = train_df_tr[features], train_df_tr[[target]]
    X_val, y_val = val_df_tr[features], val_df_tr[[target]]

    best_val_nll = np.inf
    best_residual_params = None

    for residual_params in tqdm(list(param_sampler)):

        reg = ResidualDoubleCVSafe(
            estimator=mean_model_class(**mean_params),
            estimator_resid=residual_model_class(**residual_params),        use_y_pred=use_y_pred,
            cv=get_cv_fn(target, features),
        )

        reg._fit(X_train, y_train)

        val_dist = reg._predict_proba(X_val)
        val_nll = -np.mean(val_dist.log_pdf(y_val))
        

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_residual_params = residual_params

    trainval_df = pd.concat([train_df, val_df], axis=0)

    hier_mean_transformer = HierMeanFeatureSet()
    hier_mean_transformer.fit(trainval_df)
    trainval_df_tr = hier_mean_transformer.transform_train(trainval_df)
    test_df_tr = hier_mean_transformer.transform_test(test_df)

    X_trainval, y_trainval = trainval_df_tr[features], trainval_df_tr[[target]]
    X_test, y_test = test_df_tr[features], test_df_tr[[target]]

    best_reg = ResidualDoubleCVSafe(
        estimator=mean_model_class(**mean_params),
        estimator_resid=residual_model_class(**best_residual_params),        use_y_pred=use_y_pred,
        cv=get_cv_fn(target, features),
    )
    best_reg._fit(X_trainval, y_trainval)

    test_dist = best_reg._predict_proba(X_test)
    test_pred = test_dist.mu
    test_nll = -np.mean(test_dist.log_pdf(y_test))

    all_df = pd.concat([train_df, val_df, test_df], axis=0)

    hier_mean_transformer = HierMeanFeatureSet()
    hier_mean_transformer.fit(all_df)
    all_df_tr = hier_mean_transformer.transform_train(all_df)
    X_all, y_all = all_df_tr[features], all_df_tr[[target]]

    final_reg = ResidualDoubleCVSafe(
        estimator=mean_model_class(**mean_params),
        estimator_resid=residual_model_class(**best_residual_params),
        use_y_pred=use_y_pred,
        cv=get_cv_fn(target, features),
    )
    final_reg._fit(X_all, y_all)

    return {
        "name": model_name,
        "best_residual_params": best_residual_params,
        "test_nll": test_nll,
        "test_mean_pred": test_pred,
        "reg": final_reg,
        "transfo": hier_mean_transformer
    }

def predict_winning_price_scale_model(
    train_df, val_df, test_df,
    features, target,
    mean_model_names,
    resid_model_names,
    mean_model_classes,
    resid_model_classes,
    param_grid_map,
    use_y_pred,
):  
    mean_model_summary = []
    best_mean_test_rmse = +np.inf
    best_mean_model_info = None
    for (mean_model_name, mean_model_class) in zip(mean_model_names, mean_model_classes):
        info = train_mean_model(
            train_df, val_df, test_df,
            features, target,
            mean_model_name,
            mean_model_class,
            param_grid_map[mean_model_name],
        )
        summary = {"mean_model": mean_model_name}
        summary.update(info["metrics"])
        mean_model_summary.append(summary)
        if info["metrics"]["RMSE"] < best_mean_test_rmse:
            best_mean_test_rmse = info["metrics"]["RMSE"]
            best_mean_model_info = info
    
    full_model_summary = []
    best_test_nll = +np.inf
    best_full_model_info = None
    for (resid_model_name, resid_model_class) in zip(resid_model_names, resid_model_classes):
        model_name = best_mean_model_info["name"] + "_" + resid_model_name
        info = train_scale_model(
            train_df, val_df, test_df,
            features, target,
            model_name=model_name,
            mean_model_class=best_mean_model_info["class"],
            residual_model_class=resid_model_class,
            mean_params=best_mean_model_info["best_mean_params"],
            residual_param_grid=param_grid_map[resid_model_name],
            use_y_pred=use_y_pred
        )
        test_mu = info["test_mean_pred"]
        reg = regression_metrics(test_df[[target]], test_mu)
        summary = {
            "name": info["name"],
            "MAE": reg["MAE"],
            "RMSE": reg["RMSE"],
            "R2": reg["R2"],
            "NLL": info["test_nll"],
        }
        full_model_summary.append(summary)
        if info["test_nll"] < best_test_nll:
            best_test_nll = info["test_nll"]
            best_full_model_info = {"metrics": summary}
            best_full_model_info["reg"] = info["reg"]
            best_full_model_info["transfo"] = info["transfo"]
    return best_full_model_info, mean_model_summary, full_model_summary

def train_model(df, mode, need_prep=True):
    hier_mean_feats = HierMeanFeatureSet().get_feature_names_out()

    preprocessor_mean = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES + hier_mean_feats),
        ]
    )

    preprocessor_resid = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES + hier_mean_feats + ["y_pred"]),
        ]
    )

    name_to_params_map = {
        "HM": {},
        "LR": LR_PARAM_GRID,
        "RF": RF_PARAM_GRID,
        "XGB": XGB_PARAM_GRID
    }

    name_to_class_map_mean = {
        "HM": HierMean,
        "LR": get_pipeline_wrap(preprocessor_mean, LinearRegression),
        "RF": get_pipeline_wrap(preprocessor_mean, RandomForestRegressor),
        "XGB": get_pipeline_wrap(preprocessor_mean, XGBRegressor),
    }

    name_to_class_map_resid = {
        "HM": HierMean,
        "LR": get_pipeline_wrap(preprocessor_resid, LinearRegression),
        "RF": get_pipeline_wrap(preprocessor_resid, RandomForestRegressor),
        "XGB": get_pipeline_wrap(preprocessor_resid, XGBRegressor),
    }

    if mode == "default":
        mean_model_names = ["HM"]
        resid_model_names = ["RF"]

    else:
        mean_model_names = ["HM", "RF", "XGB"]
        resid_model_names = ["RF", "XGB"]

    if need_prep:
        df = prep_df(df)

    train_df, val_df, test_df = get_train_val_test_split(df)
    best_model, mean_model_summary, full_model_summary = predict_winning_price_scale_model(
        train_df, val_df, test_df,
        features=FEATURES+hier_mean_feats+[TARGET], 
        target=TARGET,
        mean_model_names=mean_model_names,
        resid_model_names=resid_model_names,
        mean_model_classes=[name_to_class_map_mean[n] for n in mean_model_names],
        resid_model_classes=[name_to_class_map_resid[n] for n in resid_model_names],
        param_grid_map=name_to_params_map,
        use_y_pred=True
    )
    return best_model, mean_model_summary, full_model_summary

def predict(X_hat_dict, model):
    required_keys = ["investor", "quantity", "manufacturer", "country_of_origin", "province", "closing_date"]
    assert all([k in X_hat_dict for k in required_keys])
    X_hat = {k: X_hat_dict[k] for k in required_keys}
    X_hat = pd.DataFrame([X_hat])
    X_hat = prep_df(X_hat, predict=True)
    X_hat = model["transfo"].transform_test(X_hat)
    pred_dist = model["reg"]._predict_proba(X_hat)
    return pred_dist

if __name__ == "__main__":
    df = pd.read_csv("kit_test_df_clean.csv")
    mask = df.fillna("").astype(str).agg(" ".join, axis=1).apply(lambda x: ("rapid" in x.lower() or "nhanh" in x.lower()) and "hiv" in x.lower())
    df = df[mask].reset_index(drop=True)
    print(len(df))
    best_model, mean_model_summary, full_model_summary = train_model(df, mode="default", need_prep=True)
    print(best_model["metrics"])
    print(predict(df.iloc[0].to_dict(), best_model))

