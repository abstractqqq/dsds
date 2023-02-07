import polars as pl
from typing import Tuple
from scipy.stats import chi2_contingency
from concurrent.futures import ThreadPoolExecutor, as_completed

# Categorical EDA methods

def entropy_cat_eda(df:pl.DataFrame, cat:str, target:str, tot:int) -> Tuple[pl.DataFrame, str]:
    '''
        Perform entrypy categorical EDA for one variable.

        df
        cat: name of the categorical column
        target: name of the target column
    
    '''

    output = df.rename({cat:"value"})\
        .select([target, "value"])\
        .groupby(["value", target])\
        .agg([
            pl.count().alias("cnt")
        ]).with_columns(
            pl.col("cnt").sum().over("value").alias("sum_value")
        ).with_columns([
            (pl.col("cnt")/tot).alias("prob_target|value"),
            (pl.col("sum_value")/tot).alias("prob_value")
        ]).with_columns(
            (pl.col("prob_target|value")*((pl.col("prob_target|value")/pl.col("prob_value")).log())).alias("unit_entropy")
        ).with_columns([
            -pl.col("unit_entropy").sum().alias("entropy"),
            pl.col("value").cast(pl.Utf8).alias("value"),
            pl.lit(cat).alias("column")
        ]).rename({"literal":"entropy"})\
        .sort(by="value").fill_null("")

    return output, cat

def entropy_cat_eda_summary(df:pl.DataFrame, cat_columns:list[str], target:str, n_threads:int=4) -> pl.DataFrame:
    ''' 
        Perform entrypy categorical EDA for each variable in cat_columns.

        df
        cat_columns
        target: must be a binary variable. 
        n_threads: 
    '''

    tot = len(df)
    cat_dfs = []
    n = len(cat_columns)
    df2 = df.select(cat_columns + [target])
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = (ex.submit(entropy_cat_eda, df2, c, target, tot) for c in cat_columns)
        for i,f in enumerate(as_completed(futures)):
            cat_df, cat = f.result()
            cat_dfs.append(cat_df)
            print(f"Finished processing for {cat}, {i+1}/{n}.")

    return pl.concat(cat_dfs)

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
    