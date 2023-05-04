import polars as pl
import os
import numpy as np 
from typing import Tuple, Optional
from scipy.stats import chi2_contingency
from scipy.special import fdtrc
from concurrent.futures import ThreadPoolExecutor, as_completed
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

POLARS_NUMERICAL_TYPES = [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
CPU_COUNT = os.cpu_count()

def get_numeric_cols(df:pl.DataFrame, exclude:list[str]=None) -> list[str]:
    ''' 
    
    '''
    output = []
    exclude_list = [] if exclude is None else exclude
    for c,t in zip(df.columns, df.dtypes):
        if t in POLARS_NUMERICAL_TYPES and c not in exclude_list:
            output.append(c)
    return output

def get_string_cols(df:pl.DataFrame, exclude:list[str]=None) -> list[str]:
    output = []
    exclude_list = [] if exclude is None else exclude
    for c,t in zip(df.columns, df.dtypes):
        if t == pl.Utf8 and c not in exclude_list:
            output.append(c)
    return output

def _conditional_entropy(df:pl.DataFrame, target:str, predictive:str) -> pl.DataFrame:
    temp = df.groupby(predictive).agg(
        pl.count().alias("prob(predictive)")
    ).with_columns(
        pl.col("prob(predictive)") / len(df)
    )

    return df.groupby([target, predictive]).agg(
        pl.count()
    ).with_columns(
        (pl.col("count") / pl.col("count").sum()).alias("prob(target,predictive)")
    ).join(
        temp, on=predictive
    ).select([
        pl.lit(predictive).alias("feature"),
        (-((pl.col("prob(target,predictive)")/pl.col("prob(predictive)")).log() * pl.col("prob(target,predictive)")).sum()).alias("conditional_entropy")
    ])

def information_gain(df:pl.DataFrame, target:str
    , cat_cols:list[str] = None
    , top_k:int = 0
    , n_threads:int = CPU_COUNT
    , verbose:bool = True) -> pl.DataFrame:
    '''
        Computes the information gain: Entropy(target) - Conditional_Entropy(target|c), where c is a column in cat_cols.
        For more information, please take a look at https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Arguments:
            df:
            target:
            cat_cols: list of categorical columns. Note that you may use numeric columns as categorical columns provided the column has 
                a reasonably small number of distinct values.
            top_k: must be >= 0. If == 0, the entire DataFrame will be returned.
            n_threads: 4, 8 ,16 will not make any real difference. But there is a difference between 0 and 4 threads. 
            verbose: if true, print progress.
            
        Returns:
            a poalrs dataframe with information gain computed for each categorical column. 
    '''
    output = []
    cats = []
    if isinstance(cat_cols, list):
        cats.extend(cat_cols)
    else: # If cat_cols is not passed, infer it
        cats.extend(get_string_cols(df, exclude=[target]))

    if len(cats) == 0:
        print(f"No columns are provided or can be inferred.")
        print("Returned empty dataframe.")
        return pl.DataFrame()
    
    # Compute target entropy. This only needs to be done once.
    target_entropy = df.groupby(target).agg(
                        (pl.count()).alias("prob(target)") / len(df)
                    ).get_column("prob(target)").entropy() 

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = (ex.submit(_conditional_entropy, df, target, predictive) for predictive in cats)
        for i,res in enumerate(as_completed(futures)):
            ig = res.result()
            output.append(ig)
            if verbose:
                print(f"Finished processing for {cats[i]}. Progress: {i+1}/{len(cats)}")

    output = pl.concat(output).with_columns([
        pl.lit(target_entropy).alias("target_entropy"),
        (pl.lit(target_entropy) - pl.col("conditional_entropy")).alias("information_gain")
    ]).sort("information_gain", descending=True)\
        .select(["feature", "target_entropy", "conditional_entropy", "information_gain"])

    if top_k == 0:
        return output 
    else:
        return output.limit(top_k)

def f_test(df:pl.DataFrame, target:str, num_cols:list[str]=None) -> pl.DataFrame:
    '''
        Computes ANOVA one way test, the f value/score and the p value. 
        Equivalent to f_classif in sklearn.feature_selection, but is more dataframe-friendly, 
        and performs better on bigger data.

        Arguments:
            df: input Polars dataframe.
            target: the target column.
            num_cols: if provided, will run the ANOVA one way test for each column in num_cols. If none,
                will try to infer from df according to data types. Note that num_cols should be numeric!

        Returns:
            a polars dataframe with f score and p value.
    
    '''
    num_list = []
    if isinstance(num_cols, list):
        num_list.extend(num_cols)
    else:
        num_list.extend(get_numeric_cols(df, exclude=[target]))

    # Get average within group and sample variance within group.
    step_one_expr:list[pl.Expr] = [pl.count().alias("cnt")] # get cnt, and avg within classes
    step_two_expr:list[pl.Expr] = [] # Get average for each column
    step_three_expr:list[pl.Expr] = [] # Get "f score" (without some normalizer, see below)
    # Minimize the amount of looping and str concating in Python. Use Exprs as much as possible.
    for n in num_list:
        n_avg:str = n + "_avg" # avg within class
        n_tavg:str = n + "_tavg" # true avg / absolute average
        n_var:str = n + "_var" # var within class
        step_one_expr.append(
            pl.col(n).mean().alias(n_avg)
        )
        step_one_expr.append(
            pl.col(n).var(ddof=0).alias(n_var) # ddof = 0 so that we don't need to compute pl.col("cnt") - 1
        )
        step_two_expr.append( # True average of this column
            (pl.col(n_avg).dot(pl.col("cnt")) / len(df)).alias(n_tavg)
        )
        step_three_expr.append(
            # Between class var (without diving by df_btw_class) / Within class var (without dividng by df_in_class) 
            (pl.col(n_avg) - pl.col(n_tavg)).pow(2).dot(pl.col("cnt"))/ pl.col(n_var).dot(pl.col("cnt"))
        )

    # Get in class average and var
    ref = df.groupby(target).agg(step_one_expr)
    n_samples = np.float64(len(df))
    n_classes = np.float64(len(ref))
    df_btw_class = n_classes - 1 
    df_in_class = n_samples - n_classes
    
    f_values = ref.with_columns(step_two_expr).select(step_three_expr)\
            .to_numpy().ravel() * (df_in_class / df_btw_class)
    # We should scale this by (df_in_class / df_btw_class) because we did not do this earlier

    p_values = fdtrc(df_btw_class, df_in_class, f_values) # get p values 
    return pl.from_records([num_list, f_values, p_values], schema=["feature","f_value","p_value"])

# ---------------------------- BASIC STUFF ----------------------------------------------------------------

def describe(df:pl.DataFrame) -> pl.DataFrame:
    '''
        The transpose view of df.describe() for easier filtering. Add more statistics in the future (if easily
        computable.)

        Arguments:
            df:

        Returns:
            Transposed view of df.describe() with a few more interesting columns
    '''
    temp = df.describe()
    columns = temp.drop_in_place("describe")
    unique_counts = pl.Series("unique_count", df.select(
            (pl.col(c).n_unique().alias(c+"_unique_count") for c in df.columns)
        ).to_numpy().ravel())
    nums = ("count", "null_count", "mean", "std", "median", "25%", "75%")
    final = temp.transpose(include_header=True, column_names=columns).with_columns(
        (pl.col(c).cast(pl.Float64) for c in nums)
    ).with_columns(
        (pl.col("null_count")/pl.col("count")).alias("null_pct")
    ).insert_at_idx(3, unique_counts)
    
    return final.select(['column','count','null_count','null_pct','unique_count','mean','std','min','max','median','25%','75%'])

def null_removal(df:pl.DataFrame, threshold:float=0.5) -> pl.DataFrame:
    '''
        Removes columns with more than threshold% null values.

        Arguments:
            df:
            threshold:

        Returns:
            df without null_pct > threshold columns
    '''

    remove_cols = (df.null_count()/len(df)).transpose(include_header=True, column_names=["null_pct"])\
                    .filter(pl.col("null_pct") > threshold)\
                    .get_column("column").to_list()
    
    print(f"The following columns are dropped because they have more than {threshold*100:.2f}% null values. {remove_cols}.")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def var_removal(df:pl.DataFrame, threshold:float, target:str) -> pl.DataFrame:
    '''
        Removes features with low variance. Features with > threshold variance will be kept. This only works for numerical columns.
        Note that this can effectively remove (numerical) constants as well. It is, however, hard to come up with
        a uniform threshold for all features, as numerical features can be at very different scales. 

        Arguments:
            df:
            threshold:
            target: target in your predictive model. Some targets may have low variance, e.g. imbalanced binary targets. So we should exclude it.

        Returns:
            df without columns with < threshold var.
    '''
    var_expr = (pl.col(x).var() for x in get_numeric_cols(df) if x != target)
    remove_cols = df.select(var_expr).transpose(include_header=True, column_names=["var"])\
                    .filter(pl.col("var") < threshold).get_column("column").to_list()
    
    print(f"The following numeric columns are dropped because they have lower than {threshold} variance. {remove_cols}")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def get_unique_count(df:pl.DataFrame) -> pl.DataFrame:
    return df.select((
        pl.col(x).n_unique() for x in df.columns
    )).transpose(include_header=True, column_names=["n_unique"])

def constant_removal(df:pl.DataFrame, include_null:bool=True) -> pl.DataFrame:
    '''
        Removes all constant columns from dataframe.
        Arguments:
            df:
            include_null: if true, then columns with two distinct values like [value_1, null] will be considered a 
                constant column.

        Returns: 
            the df without constant columns
    '''
    temp = get_unique_count(df).filter(pl.col("n_unique") <= 2)
    remove_cols = temp.filter(pl.col("n_unique") == 1).get_column("column").to_list() # These are constants, remove.
    if include_null: 
        binary = temp.filter(pl.col("n_unique") == 2).get_column("column")
        for b in binary:
            if None in df.get_column(b).unique():
                remove_cols.append(b)

    print(f"The following columns are dropped because they are constants. {remove_cols}.")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

# ----------------------------------------------- Transformations --------------------------------------------------

def binary_encode(df:pl.DataFrame, binary_cols:list[str]=None, exclude:list[str]=None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
        The goal of this function is to map binary categorical values into [0, 1], therefore reducing the amount of encoding
        you will have to do later. This is important when you want to keep feature dimension low and when you have many binary categorical
        variables. The values will be mapped to [0, 1] by the following rule:
            if value_1 < value_2, value_1 --> 0, value_2 --> 1. E.g. 'N' < 'Y' ==> 'N' --> 0 and 'Y' --> 1
        
        In case the two distinct values are [None, value_1], and you decide to treat this variable as a binary category, then
        None --> 0 and value_1 --> 1. (If you apply constant_removal first then this column will be seen as constant and dropped.)
        
        Using one-hot-encoding will map binary categorical values to 2 columns (except when you specify drop_first=True in pd.get_dummies),
        therefore introducing unnecessary dimension. So it is better to prevent it.

        In case the distinct values the column are [null, value_1, value_2], you must first impute this column if you want this method
        to count this as a binary column. 

        Arguments:
            df:
            binary_cols: the binary_cols you wish to convert. If no input, will infer (might take time because counting unique values for each column is not cheap).
            exclude: the columns you wish to exclude in this transformation. (Only applies if you are letting the system auto-detecting binary columns.)

        Returns: 
            (the transformed dataframe, mapping table between old values to [0,1])
    '''
    mapping = {"feature":[], "to_0":[], "to_1":[], "dtype":[]}
    exprs = []
    binary_list = []
    if isinstance(binary_cols, list):
        binary_list.extend(binary_cols)
    else:
        binary_columns = get_unique_count(df).filter((pl.col("n_unique") == 2) & (~pl.col("column").is_in(exclude))).get_column("column")
        binary_list.extend(binary_columns)
    
    # Doing some repetitive operations here, but I am not sure how I can get all the data in one go.
    for b in binary_list:
        vals = df.get_column(b).unique().to_list()
        print(f"Transforming {b} into a binary column with [0, 1] ...")
        if len(vals) != 2:
            print(f"Found {b} has {len(vals)} unique values instead of 2. Not a binary variable. Ignored.")
            continue
        if vals[0] is None: # Weird code, but we need this case.
            pass
        elif vals[1] is None:
            vals[0], vals[1] = vals[1], vals[0]
        else:
            vals.sort()

        mapping["feature"].append(b)
        mapping["to_0"].append(vals[0] if vals[0] is None else str(vals[0])) # have to cast to str to avoid mixed types
        mapping["to_1"].append(str(vals[1])) # vals[1] is gauranteed to be not None by above logic
        mapping["dtype"].append("string" if isinstance(vals[1], str) else "numeric")
        
        exprs.append(
            pl.when(pl.col(b).is_null()).then(0).otherwise(
                pl.when(pl.col(b) < vals[1]).then(0).otherwise(1)
            ).cast(pl.UInt8).alias(b) 
        )

    return df.with_columns(exprs), pl.from_dict(mapping)

def get_ordinal_mapping_table(ordinal_mapping:dict[str, dict[str,int]]) -> pl.DataFrame:
    '''
        Helper function to get a table from an ordinal_mapping dict.

        Arguments:
            ordinal_mapping: {name_of_feature: {value_1 : mapped_to_number_1, value_2 : mapped_to_number_2, ...}, ...}

        Returns:
            A table with feature name, value, and mapped_to
    
    '''
    mapping_tables:list[pl.DataFrame] = []
    for feature, mapping in ordinal_mapping.items():
        count = len(ordinal_mapping[feature])
        # Python's (> 3.7) dict is ordered dict. So we can do this.
        mapping_tables.append(
            pl.from_records([[feature]*count, mapping.keys(), mapping.values()], schema=["feature","value", "mapped_to"])
        )

    return pl.concat(mapping_tables)

def ordinal_auto_encode(df:pl.DataFrame, ordinal_cols:list[str]=None, exclude:list[str]=None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
        Automatically applies ordinal encoding to the provided columns by the following logic:
            Sort the column, smallest value will be assigned to 0, second smallest will be assigned to 1...

        This will automatically detect string columns and apply this operation if ordinal_cols is not provided. 
        This method is great for string columns like age ranges, with values like ["10-20", "20-30"], etc...
        
        Arguments:
            df:
            ordinal_cols:
            exclude: the columns you wish to exclude in this transformation. (Only applies if you are letting the system auto-detecting binary columns.)
        
        Returns:
            (encoded df, mapping table)
    '''
    
    ordinal_list:list[str] = []
    if isinstance(ordinal_cols, list):
        ordinal_list.extend(ordinal_cols)
    else:
        ordinal_list.extend(get_string_cols(df, exclude=exclude))

    exprs:list[pl.Expr] = []
    ordinal_mapping:dict[str, dict[str,int]] = {}
    rename_dict:dict[str, str] = {}
    for c in ordinal_list:
        sorted_uniques = df.get_column(c).unique().sort()
        count = len(sorted_uniques)
        mapping:dict[str, int] = dict(zip(sorted_uniques, range(count)))
        ordinal_mapping[c] = mapping
        exprs.append(pl.col(c).map_dict(mapping).cast(pl.UInt32))
        rename_dict[c] = c + "_ordinal"

    return df.with_columns(exprs).rename(rename_dict), get_ordinal_mapping_table(ordinal_mapping)

def ordinal_encode(df:pl.DataFrame, ordinal_mapping:dict[str, dict[str,int]], default:int = -1) -> pl.DataFrame:
    '''
        Ordinal encode the data with given mapping.
        
        Notice that it is assumed that you already have the mapping,
        since you have to supply the ordinal_mapping argument. If you still want the tabular output format,
        please call get_ordinal_mapping_table with ordinal_mapping, which will create a table from this.

        Arguments:
            df:
            ordinal_mapping:
            default: if a value for a feature does not exist in ordinal_mapping, use default.

        Returns:
            encoded df
    '''
    
    rename_dict:dict[str, str] = {}
    exprs:list[pl.Expr] = []
    for c in ordinal_mapping:
        if c in df.columns:
            mapping = ordinal_mapping[c]
            exprs.append(pl.col(c).map_dict(mapping, default=default).cast(pl.UInt32))
            rename_dict[c] = c + "_ordinal"
        else:
            print(f"Found that column {c} is not in df. Skipped.")

    return df.with_columns(exprs).rename(rename_dict)

def percentile_binning(df:pl.DataFrame, num_cols:list[str]=None, exclude:list[str]=None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
        Bin your continuous variable X into X_percentiles. This will create at most 100 + 1 bins, where each percentile could
        potentially be a bin and null will be mapped to bin = 0. Bin 1 means percentile 0 to 1. Generally, bin X groups the
        population from bin X-1 to bin X into one bucket.

        I see some potential optimization opportunities here.

        Arguments:
            df:
            num_cols: 
            exclude:

        Returns:
            (A transformed dataframe, a mapping table (value to percentile))
    
    '''

    # Percentile Binning

    num_list:list[str] = []
    exclude_list:list[str] = [] if exclude is None else exclude
    if isinstance(num_cols, list):
        num_list.extend(num_cols)
    else:
        num_list.extend(get_numeric_cols(df, exclude=exclude_list))

    tables:list[pl.DataFrame] = []
    exprs:list[pl.Expr] = []
    rename_dict:dict[str,str] = {}
    for c in num_list:
        percentile = df.groupby(c).agg(pl.count().alias("cnt"))\
            .sort(c)\
            .with_columns(
                ((pl.col("cnt").cumsum()*100)/len(df)).ceil().alias("percentile")
            ).groupby("percentile")\
            .agg([
                pl.col(c).min().alias("min"),
                pl.col(c).max().alias("max"),
                pl.col(c).sum().alias("cnt"),
            ]).sort("percentile").select([
                pl.lit(c).alias("feature"),
                pl.col("percentile").cast(pl.UInt8),
                "min",
                "max",
                "cnt",
            ])
        
        first_row = percentile.select(["percentile","min", "max"]).to_numpy()[0, :] # First row
        # Need to handle an extreme case when percentile looks like 
        # percentile   min   max
        #  p1         null  null
        #  p2          ...   ...
        if np.isnan(first_row[2]):
            print(f"For feature {c}, we found (percentile, min, max) = {first_row}, which cannot be correctly processed by the algorithm. Skipped.")
            continue 

        temp_df = df.lazy().filter(pl.col(c).is_not_null()).sort(c)\
            .join_asof(other=percentile.lazy(), left_on=c, right_on="max", strategy="forward")\
            .select([c, "percentile"])\
            .unique().collect()
        
        mapping = dict(zip(temp_df[c], temp_df["percentile"]))

        exprs.append(
            pl.col(c).map_dict(mapping, default=0).cast(pl.UInt8)
        )
        rename_dict[c] = c + "_percentile"
        percentile = percentile.with_columns([
            pl.col("min").cast(pl.Float32),
            pl.col("max").cast(pl.Float32),
            pl.col("cnt").cast(pl.UInt32)
        ]) # Need to do this because we need a uniform format in order to stack these columns.
        tables.append(percentile)

    return df.with_columns(exprs).rename(rename_dict), pl.concat(tables)

# --------------------- Other, miscellaneous helper functions ----------------------------------------------

def get_numpy(df:pl.DataFrame, target:str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    '''
        Create NumPy matrices/array for feature matrix X and target y. 
        
        Note that this implementation will "consume df" column by column, thus saving memory used in the process.
        If memory is not a problem, do directly df.select(feature).to_numpy(). IMPORTANT: df will be deleted at the end.

    
    '''
    features:list[str] = df.columns 
    features.remove(target)
    y = df.drop_in_place(target).to_numpy().ravel() 
    columns = []
    for c in features:
        columns.append(
            df.drop_in_place(c).to_numpy()
        )

    del df
    X = np.concatenate(columns, axis=1)
    return X,y,features


