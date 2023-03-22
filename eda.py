import polars as pl
import os 
from typing import Tuple
from scipy.stats import chi2_contingency
from concurrent.futures import ThreadPoolExecutor, as_completed

# Categorical EDA methods

def _information_gain(df:pl.DataFrame, target:str, predictive:str) -> pl.DataFrame:
    temp = df.groupby([predictive]).agg([
        pl.count().alias("prob(predictive)")
    ]).with_columns([
        pl.col("prob(predictive)") / len(df)
    ])

    target_entropy = df.groupby([target]).agg([
        pl.count().alias("prob(target)")
    ]).with_columns([
        pl.col("prob(target)") / len(df)
    ]).select(pl.col("prob(target)").entropy()).to_numpy()[0,0]

    return df.groupby([target, predictive]).agg([
        pl.count()
    ]).with_columns([
        (pl.col("count") / pl.col("count").sum()).alias("prob(target,predictive)")
    ]).join(
        temp, on=predictive
    ).select([
        pl.lit(predictive).alias("Predictive Variable"),
        pl.lit(target_entropy).alias("Target Entropy"),
        (-((pl.col("prob(target,predictive)")/pl.col("prob(predictive)")).log() * pl.col("prob(target,predictive)")).sum()).alias("Conditional Entropy")
    ]).with_columns(
        (pl.col("Target Entropy") - pl.col("Conditional Entropy")).alias("Information Gain")
    )

def information_gain(df:pl.DataFrame, target:str, cat_cols:list[str]=None) -> pl.DataFrame:
    output = []
    cats = []
    if cat_cols:
        cats.extend(cat_cols)
    else: # If cat_cols is not passed, infer it
        for c,t in zip(df.columns, df.dtypes):
            if t == pl.Utf8 and c != target:
                cats.append(c)

    if len(cats) == 0:
        return pl.DataFrame() 

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = ( ex.submit(_information_gain, df, target, predictive) for predictive in cats )
        for i,res in enumerate(as_completed(futures)):
            ig = res.result()
            print(f"Finished processing for {cats[i]}. {i+1}/{len(cats)}")
            output.append(ig)

    return pl.concat(output)

def get_contigency_table(df:pl.DataFrame, a:str, b:str) -> pl.DataFrame:
    '''
        df
        a is values (utf8), the row in the contigency table
        b is target, the columns in the contigency table
    
    '''

    return df.rename({a:"value"})\
        .select([pl.col(b).fill_null(b+"_UNK"), pl.col("value").fill_null("_UNK")])\
        .groupby(["value", b])\
        .agg([
            pl.count().alias("cnt")
        ]).pivot(columns="value", values="cnt", index=b)\
            .fill_null(0)
            

def chi2_contigency_test(df:pl.DataFrame, cat:str, target:str) -> Tuple[str, object]:
    '''
        Transforms data into a contigency table and perform chi squared test on it to see if 
        cat and target are independent or not.



        returns
            (cat, a class that contains chi2_contigency_test results)
    
    '''
    
    temp = get_contigency_table(df, cat, target)
    cat_values = temp.columns
    cat_values.remove(target)
    res = chi2_contingency(temp.select(cat_values).to_numpy(), lambda_ = "log-likelihood")
    return cat, res

def chi2_contigency_summary(df:pl.DataFrame, cat_columns:list[str], target:str
    , threshold:float=0.05, n_threads:int = 4
) -> pl.DataFrame:
    '''
    
    
    '''
    n = len(cat_columns)
    final_results = []
    df2 = df.select(cat_columns + [target])
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = (ex.submit(chi2_contigency_test, df2, c, target) for c in cat_columns)
        for i,f in enumerate(as_completed(futures)):
            cat, res = f.result()
            final_results.append((cat, res.pvalue, res.pvalue < threshold))
            print(f"Finished processing for {cat}, {i+1}/{n}.")

    return pl.from_records(final_results, schema=["feature_name", "p-value", "is_dependent"])
    