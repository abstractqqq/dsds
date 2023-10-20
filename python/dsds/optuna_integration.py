import lightgbm as lgb
import polars as pl
import optuna
import dsds
import dsds.sample as sa
import dsds.metrics as me
import dsds.prescreen as ps
from .type_alias import (
    BinaryMetrics
    , PolarsFrame
)
from typing import Any, Optional, Tuple

def suggest_b_lgbm_hyperparams(
    df:PolarsFrame
    , target:str
    , max_trials: int = 50
    , metric: BinaryMetrics = "log_loss"
    , sample_frac: Optional[float] = None
    , timeout:int = 60 * 10
) -> Tuple[dict[str, Any], optuna.study.Study]:
    '''
    A quick method to suggest hyperparameters for Binary LGBM model. This is used internally in
    feature selections. More specifically, the hyperparameters suggested here will be used in 
    feature selection algorithms when model-based feature importance is needed. It can also be used
    as a standalone hyperparamter tuner. It is recommended that you sample the data if data is huge.
    
    This function always returns a 2-tuple of (dict of best params, Optuna Study)
    
    Parameters
    ----------
    df
        Either a lazy or an eager Polars dataframe. This method will internally collect.
    target
        Training target. Only binary.
    max_trials
        The maximum numbner of Optuna trials to run.
    sample_frac
        Optional. If provided, will try to sample this fraction of the data from input.
    timeout
        Time out for the Optuna study in seconds
    '''

    if sample_frac is None:
        df_local = df.lazy().collect()
    else:
        if isinstance(df, pl.LazyFrame):
            df_local = sa.lazy_sample(df, sample_frac=sample_frac).collect()
        else:
            df_local = df.sample(fraction=sample_frac)

    if not ps.check_binary_target(df_local, target):
        raise ValueError("Target must be binary, represented by 0s and 1s.")

    if metric == "log_loss":
        metric_func = me.log_loss
        metric_word = "binary_logloss"
        direction = "minimize"
    elif metric == "auc":
        metric_func = me.roc_auc
        metric_word = "auc"
        direction = "maximize"
    elif metric == "brier_loss":
        metric_func = me.mse
        metric_word = "mse"
        direction = "minimize"

    def objective(trial):

        train, valid = sa.train_test_split(df_local, train_frac=0.75, collect=True)
        train_y = train.drop_in_place(target).to_numpy()
        valid_y = valid.drop_in_place(target).to_numpy()
        train = train.to_numpy()
        valid = valid.to_numpy()

        dtrain = lgb.Dataset(train, label=train_y)
        dvalid = lgb.Dataset(valid, label=valid_y)
        param = {
            "objective": "binary",
            "metric": metric_word,
            "verbosity": 0,
            "max_depth":trial.suggest_int("max_depth", 2,16),
            "num_iterations": trial.suggest_int("num_iterations", 50, 200),
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, metric_word)
        lgbm_ = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])
        preds = lgbm_.predict(valid)
        return metric_func(valid_y, preds)

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        , direction=direction
    )

    study.optimize(objective, n_trials=max_trials, n_jobs=dsds.THREADS, timeout=timeout, gc_after_trial=True)
    trial = study.best_trial
    print(f"Best params: {trial.params}.\nFound at trial: {trial.number}.\nTime took: {trial.duration.seconds}s.")
    return study.best_params, study