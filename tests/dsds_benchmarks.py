import pytest
import polars as pl
import pandas as pd
import dsds.sample as sa
import dsds.metrics as me
import dsds.transform as t
import dsds.fs as fs
import dsds.encoders as enc
import numpy as np
from polars.testing import assert_frame_equal
from dsds.type_alias import PolarsFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.datasets import fetch_openml, make_classification
from category_encoders import TargetEncoder, OneHotEncoder, WOEEncoder
from mrmr import mrmr_classif
# from category_encoders import target_encoder, WOEEncoder, OneHotEncoder

set_config(transform_output = "pandas")

def dsds_train_test_split(df: PolarsFrame, train_frac:float = 0.75) -> tuple[pl.DataFrame, pl.DataFrame]:
    return sa.train_test_split(df, train_frac=train_frac)

def sklearn_train_test_split(df: pd.DataFrame, train_frac:float = 0.75) -> list:
    X_train, X_test = train_test_split(df, train_size=train_frac)
    return X_train, X_test

def smape_numpy(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))/2)

@pytest.fixture
def encoder_test() -> pl.DataFrame:
    data = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    use = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    df = data.data[use].copy()
    df["target"] = [1 if x > 200000 else 0 for x in data.target]
    df = pd.concat([df.copy()]*50)
    return pl.from_pandas(df) # 73k rows

@pytest.fixture
def pldf_with_nulls() -> pl.DataFrame:

    c1 = list(range(500_000))
    c2 = list(range(500_000, 1_000_000))
    c3 = list(range(1_000_000, 1_500_000))
    for i in range(0, 500_000, 2000):
        c1[i] = None
        c2[i] = None
        c3[i] = None

    return pl.DataFrame({
        "c1": c1,
        "c2": c2,
        "c3": c3
    })

@pytest.fixture
def pddf_with_nulls() -> pl.DataFrame:

    c1 = list(range(500_000))
    c2 = list(range(500_000, 1_000_000))
    c3 = list(range(1_000_000, 1_500_000))
    for i in range(0, 500_000, 2000):
        c1[i] = None
        c2[i] = None
        c3[i] = None

    return pd.DataFrame({
        "c1": c1,
        "c2": c2,
        "c3": c3
    })

@pytest.mark.benchmark(group="impute_median")
def test_impute_median_500k_dsds(pldf_with_nulls, benchmark):

    df: pl.DataFrame = pldf_with_nulls
    imputed_pl = benchmark(
        t.impute, df, ["c1", "c2", "c3"]
    )

    si = SimpleImputer(strategy="median")
    imputed_pd = pl.from_pandas(si.fit_transform(df.to_pandas()))
    assert_frame_equal(imputed_pl, imputed_pd)

@pytest.mark.benchmark(group="impute_median")
def test_impute_median_500k_sklearn(pddf_with_nulls, benchmark):

    df: pd.DataFrame = pddf_with_nulls
    si = SimpleImputer(strategy="median")
    _ = benchmark(
        si.fit_transform, df
    ) # The fact that they are the same is tested in test_impute_median_500k_dsds
    assert 1 == 1

@pytest.mark.benchmark(group="standard_scaling")
def test_scale_500k_dsds(pldf_with_nulls, benchmark):

    df: pl.DataFrame = pldf_with_nulls
    scaled_pl = benchmark(
        t.scale, df, ["c1", "c2", "c3"]
    )

    ss = StandardScaler()
    scaled_pd = pl.from_pandas(ss.fit_transform(df.to_pandas()))
    assert_frame_equal(scaled_pl, scaled_pd)

@pytest.mark.benchmark(group="standard_scaling")
def test_scale_500k_sklearn(pddf_with_nulls, benchmark):

    df: pd.DataFrame = pddf_with_nulls
    ss = StandardScaler()
    _ = benchmark(
        ss.fit_transform, df
    ) # The fact that they are the same is tested in test_scale_500k_dsds
    assert 1 == 1


@pytest.mark.benchmark(group="mape")
def test_mape_200k_dsds(benchmark):
    predicted = 1 + np.random.random(size=200_000)
    actual = 1 + np.random.random(size=200_000)

    dsds_res = benchmark(
        me.mape, actual, predicted
    )

    sklearn_res = round(mean_absolute_percentage_error(actual, predicted), 12)
    assert round(dsds_res, 12) == sklearn_res

@pytest.mark.benchmark(group="mape")
def test_mape_200k_sklearn(benchmark):
    predicted = 1 + np.random.random(size=200_000)
    actual = 1 + np.random.random(size=200_000)

    sklearn_res = benchmark(
        mean_absolute_percentage_error, actual, predicted
    )

    dsds_res = round(mean_absolute_percentage_error(actual, predicted), 12)
    assert round(sklearn_res, 12) == dsds_res

@pytest.mark.benchmark(group="smape")
def test_smape_500k_dsds(benchmark):
    predicted = 1 + np.random.random(size=500_000)
    actual = 1 + np.random.random(size=500_000)

    dsds_res = benchmark(
        me.smape, actual, predicted
    )

    numpy_res = smape_numpy(actual, predicted)
    assert np.isclose(dsds_res, numpy_res)

@pytest.mark.benchmark(group="smape")
def test_smape_500k_numpy(benchmark):
    predicted = 1 + np.random.random(size=500_000)
    actual = 1 + np.random.random(size=500_000)

    numpy_res = benchmark(
        smape_numpy, actual, predicted
    )

    dsds_res = me.smape(actual, predicted)
    assert np.isclose(dsds_res, numpy_res)

@pytest.mark.benchmark(group="logloss")
def test_logloss_200k_dsds(benchmark):
    predicted = np.random.random(size=200_000)
    actual = np.round(np.random.random(size=200_000)).astype(np.int8)

    dsds_res = benchmark(
        me.logloss, actual, predicted
    )

    sklearn_res = round(log_loss(actual, predicted), 12)
    assert round(dsds_res,12) == sklearn_res

@pytest.mark.benchmark(group="logloss")
def test_logloss_200k_sklearn(benchmark):
    predicted = np.random.random(size=200_000)
    actual = np.round(np.random.random(size=200_000)).astype(np.int8)

    sklearn_res = benchmark(
        log_loss, actual, predicted
    )

    dsds_res = round(me.logloss(actual, predicted), 12)
    assert dsds_res == round(sklearn_res, 12)

@pytest.mark.benchmark(group="roc_auc")
def test_roc_auc_200k_dsds(benchmark):
    predicted = np.random.random(size=200_000)
    actual = np.round(np.random.random(size=200_000)).astype(np.int8)

    dsds_res = benchmark(
        me.roc_auc, actual, predicted
    )

    sklearn_res = round(roc_auc_score(actual, predicted), 12)
    assert round(dsds_res, 12) == sklearn_res

@pytest.mark.benchmark(group="roc_auc")
def test_roc_auc_200k_sklearn(benchmark):
    predicted = np.random.random(size=200_000)
    actual = np.round(np.random.random(size=200_000)).astype(np.int8)

    sklearn_res = benchmark(
        roc_auc_score, actual, predicted
    )

    dsds_res = round(me.roc_auc(actual, predicted), 12)
    assert dsds_res == round(sklearn_res, 12)

@pytest.mark.benchmark(group="train_test_split")
def test_train_test_split_on_2mm_dsds(benchmark):
    df = pl.DataFrame({
        "a": range(2_000_000),
        "b": range(2_000_000),
        "c": range(2_000_000),
        "d": range(2_000_000),
        "e": range(2_000_000),
        "f": range(2_000_000),
        "g": range(2_000_000)
    })
    _ = benchmark(
        dsds_train_test_split, df, train_frac=0.75
    )
    train, test = dsds_train_test_split(df, train_frac=0.75)
    assert round(len(train)/len(df),2) == 0.75
    assert round(len(test)/len(df),2) == 0.25

@pytest.mark.benchmark(group="train_test_split")
def test_train_test_split_on_2mm_sklearn(benchmark):
    df = pl.DataFrame({
        "a": range(2_000_000),
        "b": range(2_000_000),
        "c": range(2_000_000),
        "d": range(2_000_000),
        "e": range(2_000_000),
        "f": range(2_000_000),
        "g": range(2_000_000)
    })
    _ = benchmark(
        sklearn_train_test_split, df, train_frac=0.75
    )
    train, test = sklearn_train_test_split(df, train_frac=0.75)
    assert round(len(train)/len(df),2) == 0.75
    assert round(len(test)/len(df),2) == 0.25

@pytest.mark.benchmark(group="woe_encoder")
def test_woe_encoding_74k_dsds(encoder_test, benchmark):

    to_be_encoded = ["MSZoning", 'CentralAir', 'Heating']
    df = encoder_test

    encoded = benchmark(
        enc.woe_cat_encode, df, "target", to_be_encoded
    )

    df_pd = df.to_pandas()
    woe = WOEEncoder(cols=to_be_encoded)
    ce_result = woe.fit_transform(X=df_pd, y=df_pd["target"])

    assert_frame_equal(
        encoded,
        pl.from_pandas(ce_result)
    )

@pytest.mark.benchmark(group="woe_encoder")
def test_woe_encoding_74k_category_encoders(encoder_test, benchmark):

    to_be_encoded = ["MSZoning", 'CentralAir', 'Heating']
    df = encoder_test.to_pandas()

    woe = WOEEncoder(cols=to_be_encoded)
    _ = benchmark(
        woe.fit_transform, df, df["target"]
    )
    # Equality is tested in test_woe_encoding_74k_dsds
    assert True 

@pytest.mark.benchmark(group="target_encoder")
def test_target_encoding_74k_dsds(encoder_test, benchmark):

    to_be_encoded = ["MSZoning", 'CentralAir', 'Heating']
    df = encoder_test

    encoded = benchmark(
        enc.smooth_target_encode, df, "target", to_be_encoded, 20, 10
    )

    df_pd = df.to_pandas()
    target = TargetEncoder(cols=to_be_encoded, min_samples_leaf=20, smoothing=10)
    ce_result = target.fit_transform(X=df_pd, y=df_pd["target"])

    assert_frame_equal(
        encoded,
        pl.from_pandas(ce_result)
    )

@pytest.mark.benchmark(group="target_encoder")
def test_target_encoding_74k_category_encoders(encoder_test, benchmark):

    to_be_encoded = ["MSZoning", 'CentralAir', 'Heating']
    df = encoder_test.to_pandas()

    target = TargetEncoder(cols=to_be_encoded, min_samples_leaf=20, smoothing=10)
    _ = benchmark(
        target.fit_transform, df, df["target"]
    )
    # Equality is tested in test_woe_encoding_74k_dsds
    assert True

@pytest.fixture
def mrmr_pl_df() -> pl.DataFrame:
    orig_x, orig_y = make_classification(n_samples = 50_000, n_features = 500, n_informative = 60, n_redundant = 440)
    df_pl = pl.from_numpy(orig_x).insert_at_idx(0, pl.Series("target", orig_y))
    return df_pl

def mrmr_package(df:pd.DataFrame, target:str, k:int) -> list[str]:
    features = list(df.columns)
    features.remove(target)
    X = df[features]
    y = df[target]
    output = mrmr_classif(X, y, K = k)
    return output

@pytest.mark.benchmark(group="MRMR")
def test_mrmr_dsds(mrmr_pl_df, benchmark):
    
    result = benchmark(
        fs.mrmr, mrmr_pl_df, "target", 50
    )
    result2 = mrmr_package(mrmr_pl_df.to_pandas(), "target", 50)
    assert result == result2

@pytest.mark.benchmark(group="MRMR")
def test_mrmr_package(mrmr_pl_df, benchmark):

    df_pd = mrmr_pl_df.to_pandas()
    _ = benchmark(
        mrmr_package, df_pd, "target", 50
    )
    assert True