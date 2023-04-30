import polars as pl
import os
from typing import Tuple, Optional
from scipy.stats import chi2_contingency
from scipy.special import fdtrc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Categorical EDA methods

def _conditional_entropy(df:pl.DataFrame, target:str, predictive:str) -> pl.DataFrame:
    temp = df.groupby([predictive]).agg([
        pl.count().alias("prob(predictive)")
    ]).with_columns([
        pl.col("prob(predictive)") / len(df)
    ])

    return df.groupby([target, predictive]).agg([
        pl.count()
    ]).with_columns([
        (pl.col("count") / pl.col("count").sum()).alias("prob(target,predictive)")
    ]).join(
        temp, on=predictive
    ).select([
        pl.lit(predictive).alias("Predictive Variable"),
        (-((pl.col("prob(target,predictive)")/pl.col("prob(predictive)")).log() * pl.col("prob(target,predictive)")).sum()).alias("Conditional Entropy")
    ])

def information_gain(df:pl.DataFrame, target:str
    , cat_cols:list[str]=None
    , n_threads:int=os.cpu_count()) -> pl.DataFrame:
    '''
        Computes the information gain: Entropy(target) - Conditional_Entropy(target | c), where c is a column in cat_cols.
        For more information, please take a look at https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Arguments:
            df:
            target:
            cat_cols: Note that you may use numeric columns as categorical columns provided the column has 
                a reasonably small number of distinct values.
            n_threads: 4, 8 ,16 will not make any real difference. But 0 there is a difference between 0 and 4 threads. 
        
        Returns:
            a poalrs dataframe with information gain computed for each categorical column. 

    '''
    output = []
    cats = []
    if isinstance(cat_cols, list):
        cats.extend(cat_cols)
    else: # If cat_cols is not passed, infer it
        for c,t in zip(df.columns, df.dtypes):
            if t == pl.Utf8 and c != target:
                cats.append(c)

    if target in cats:
        cats.remove(target)

    if len(cats) == 0 or not (target in df.columns):
        print(f"No columns are provided or can be inferred, or {target} is not in input df.")
        print("Returned empty dataframe.")
        return pl.DataFrame()
    
    # Compute target entropy. This only needs to be done once.
    target_entropy = df.groupby([target]).agg([
                        pl.count().alias("prob(target)")
                    ]).with_columns([
                        pl.col("prob(target)") / len(df)
                    ]).select(pl.col("prob(target)").entropy())\
                        .to_numpy()[0,0]

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = ( ex.submit(_conditional_entropy, df, target, predictive) for predictive in cats )
        for i,res in enumerate(as_completed(futures)):
            ig = res.result()
            output.append(ig)
            print(f"Finished processing for {cats[i]}. {i+1}/{len(cats)}")

    return pl.concat(output).with_columns([
        pl.lit(target_entropy).alias("Target Entropy"),
        (pl.lit(target_entropy) - pl.col("Conditional Entropy")).alias("Information Gain")
    ])

def f_score(df:pl.DataFrame, target:str, num_cols:list[str]=None) -> pl.DataFrame:
    '''
        Computes ANOVA one way test, the f value/score and the p value. 
        Equivalent to f_classif in sklearn.feature_selection, but is more dataframe-friendly, 
        and performs better on bigger data. (Because NumPy is not parallelized while Polars will use all
        available CPUs on the machine.)

        Arguments:
            df: input Polars dataframe.
            target: the target column.
            num_cols: if provided, will run the ANOVA one way test for each column in num_cols. If none,
                will try to infer from df according to data types. Note that num_cols should be numeric!

        Returns:
            A polars dataframe with f score and p value.
    
    '''
    
    num_list = []
    if isinstance(num_cols, list):
        num_list.extend(num_cols)
    else:
        for c,t in zip(df.columns, df.dtypes):
            if t != pl.Utf8 and t != pl.Struct and c != target: # Improve this in the future
                num_list.append(c)
    
    step_one_expr:list[pl.Expr] = [pl.count().alias("cnt")] # Get average within group and sample variance within group.
    step_two_expr:list[pl.Expr] = [] # Get true average for each column
    step_three_expr:list[pl.Expr] = [] # Get "f score" (without some normalizer, see below)
    for n in num_list:
        n_avg = n + "_avg"
        n_tavg = n + "_tavg"
        n_var = n + "_var"
        step_one_expr.append(
            pl.col(n).mean().alias(n_avg)
        )
        step_one_expr.append(
            pl.col(n).var(ddof=0).alias(n_var)
        )
        step_two_expr.append(
            # True average of this column
            (pl.col(n_avg).dot(pl.col("cnt")) / len(df)).alias(n_tavg)
        )
        step_three_expr.append(
            # Between class var (without diving by df_btw_class) / Within class var (without dividng by df_in_class) 
            (pl.col(n_avg) - pl.col(n_tavg)).pow(2).dot(pl.col("cnt"))/ pl.col(n_var).dot(pl.col("cnt"))
        )

    # Get in class average and var
    ref = df.groupby(target).agg(step_one_expr)
    n_samples = len(df)
    n_classes = len(ref)
    df_btw_class = n_classes - 1 
    df_in_class = n_samples - n_classes
    
    f_scores = ref.with_columns(step_two_expr).select(step_three_expr)\
            .to_numpy().ravel() * (df_in_class / df_btw_class)
    # We should scale this by (df_in_class / df_btw_class)

    p_values = fdtrc(df_btw_class, df_in_class, f_scores) # get p values 
    return pl.from_records([num_list, f_scores, p_values], schema=["feature","f_score","p_value"])


# def get_contigency_table(df:pl.DataFrame, a:str, b:str) -> pl.DataFrame:
#     '''
#         df
#         a is values (utf8), the row in the contigency table
#         b is target, the columns in the contigency table
    
#     '''

#     return df.rename({a:"value"})\
#         .select([pl.col(b).fill_null(b+"_UNK"), pl.col("value").fill_null("_UNK")])\
#         .groupby(["value", b])\
#         .agg([
#             pl.count().alias("cnt")
#         ]).pivot(columns="value", values="cnt", index=b)\
#             .fill_null(0)
            

# def chi2_contigency_test(df:pl.DataFrame, cat:str, target:str) -> Tuple[str, object]:
#     '''
#         Transforms data into a contigency table and perform chi squared test on it to see if 
#         cat and target are independent or not.



#         returns
#             (cat, a class that contains chi2_contigency_test results)
    
#     '''
    
#     temp = get_contigency_table(df, cat, target)
#     cat_values = temp.columns
#     cat_values.remove(target)
#     res = chi2_contingency(temp.select(cat_values).to_numpy(), lambda_ = "log-likelihood")
#     return cat, res

# def chi2_contigency_summary(df:pl.DataFrame, cat_columns:list[str], target:str
#     , threshold:float=0.05, n_threads:int = 4
# ) -> pl.DataFrame:
#     '''
    
    
#     '''
#     n = len(cat_columns)
#     final_results = []
#     df2 = df.select(cat_columns + [target])
#     with ThreadPoolExecutor(max_workers=n_threads) as ex:
#         futures = (ex.submit(chi2_contigency_test, df2, c, target) for c in cat_columns)
#         for i,f in enumerate(as_completed(futures)):
#             cat, res = f.result()
#             final_results.append((cat, res.pvalue, res.pvalue < threshold))
#             print(f"Finished processing for {cat}, {i+1}/{n}.")

#     return pl.from_records(final_results, schema=["feature_name", "p-value", "is_dependent"])

# ---------------------------- BASIC STUFF ----------------------------------------------------------------

def null_removal(df:pl.DataFrame, threshold:float=0.5) -> pl.DataFrame:
    '''
        Removes columns with more than threshold% null values.

        Arguments:
            df:
            threshold:

        Returns:
            the df without those columns
    '''

    remove_cols = (df.null_count()/len(df)).transpose(include_header=True, column_names=["null_pct"])\
                    .filter(pl.col("null_pct") > threshold)\
                    .get_column("column").to_list()
    
    print(f"The following columns are dropped because they have more than {threshold*100:.2f}% null values. {remove_cols}.")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def get_unique_count(df:pl.DataFrame) -> pl.DataFrame:
    return df.select([
        pl.col(x).n_unique() for x in df.columns
    ]).transpose(include_header=True, column_names=["n_unique"])

def constant_removal(df:pl.DataFrame, include_null:bool=True) -> pl.DataFrame:
    '''
        Removes all constant columns from dataframe.
        If include_null = True, then if a column has two distinct values {value_1, null}, then this will be considered a 
        constant column.

        Arguments:
            df:
            include_null: 

        Returns: 
            the df without constant columns
    '''

    temp = get_unique_count(df).filter(pl.col("n_unique") <= 2)
    remove_cols = []
    constants = temp.filter(pl.col("n_unique") == 1).get_column("column").to_list()
    remove_cols.extend(constants)

    if include_null: 
        binary = temp.filter(pl.col("n_unique") == 2).get_column("column").to_list()
        for b in binary:
            unique_values = df.get_column(b).unique().to_list()
            if None in unique_values:
                remove_cols.append(b)

    print(f"The following columns are dropped because they are constants. {remove_cols}.")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def binary_transform(df:pl.DataFrame, binary_cols:list[str]=None, exclude:list[str]=None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
        The goal of this function is to map binary categorical values into [0, 1], therefore reducing the amount of encoding
        you will have to do later. This is important when you want to keep feature dimension low when you have many binary categorical
        variables. The values will be mapped to [0, 1] by the following rule:
            if value_1 < value_2, value_1 --> 0, value_2 --> 1. E.g. 'N' < 'Y' ==> 'N' --> 0 and 'Y' --> 1
        
        In case the two distinct values are [None, value_1], and you decide to treat this variable as a binary category, then
        None --> 0 and value_1 --> 1.
        
        Using one-hot-encoding will map binary categorical values to 2 columns (except when you specify drop_first=True in pd.get_dummies),
        therefore introducing unnecessary dimension. So it is better to prevent it. 

        binary_cols: the binary_cols you wish to convert. If no input, will infer (might take time because counting unique values for each column is not cheap).
        exclude: the columns you wish to exclude in this transformation.

        Arguments:
            df:
            binary_cols:
            exclude: 

        Returns: 
            (the transformed dataframe, mapping table between old values to [0,1])
    '''
    mapping = {"feature":[], "to_0":[], "to_1":[], "dtype":[]}
    exprs = []
    binary_list = []
    if binary_cols is None:
        binary_columns = get_unique_count(df).filter(pl.col("n_unique") == 2).get_column("column").to_list()

        print(f"Found the following binary columns: {binary_columns}.")
        binary_list.extend(binary_columns)
    
    
    exclude_list = [] if exclude is None else exclude.copy()
    # Doing some repetitive operations here, but I am not sure how I can get all the data in one go.
    for b in binary_list:
        if b in exclude_list:
            print(f"Found {b} is in exclude list. Ignored.")
        else:
            vals = df.get_column(b).unique().to_list()
            if vals[0] is None: # Weird code, but we need this case.
                pass 
            elif vals[1] is None:
                vals[0], vals[1] = vals[1], vals[0]
            else:
                vals.sort()

            mapping["feature"].append(b)
            mapping["to_0"].append(vals[0] if vals[0] is None else str(vals[0])) # have to cast to str to avoid mixed types
            mapping["to_1"].append(vals[1] if vals[1] is None else str(vals[1]))
            mapping["dtype"].append("string" if isinstance(vals[1], str) else "numeric")
            
            exprs.append(
                pl.when(pl.col(b).is_null()).then(0).otherwise(
                    pl.when(pl.col(b) < vals[1]).then(0).otherwise(1)
                ).alias(b) 
            )

    return df.with_columns(exprs), pl.from_dict(mapping)


