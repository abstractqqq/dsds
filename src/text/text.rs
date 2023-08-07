use crate::snowball::{SnowballEnv, algorithms};
use crate::text::consts::EN_STOPWORDS;
use polars_lazy::dsl::GetOutput;
use rayon::prelude::*;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::PyDataFrame;
use std::iter::zip;

// fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
//     if n == 1 {
//         vec![(0, len)]
//     } else {
//         let chunk_size = len / n;

//         (0..n)
//             .map(|partition| {
//                 let offset = partition * chunk_size;
//                 let len = if partition == (n - 1) {
//                     len - offset
//                 } else {
//                     chunk_size
//                 };
//                 (partition * chunk_size, len)
//             })
//             .collect()
//     }
// }

#[pyfunction]
pub fn rs_cnt_vectorizer(
    pydf: PyDataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc:u32
    , max_feautures: u32
    , lowercase: bool
) -> PyResult<PyDataFrame> {

    let df: DataFrame = pydf.into();
    let df: DataFrame = count_vectorizer(df, c, stemmer, min_dfreq, max_dfreq, max_word_per_doc, max_feautures, lowercase)
                        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(df))

}

#[pyfunction]
pub fn rs_tfidf_vectorizer(
    pydf: PyDataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc:u32
    , max_feautures: u32
    , lowercase: bool
) -> PyResult<PyDataFrame> {

    let df: DataFrame = pydf.into();
    let df: DataFrame = tfidf_vectorizer(df, c, stemmer, min_dfreq, max_dfreq, max_word_per_doc, max_feautures, lowercase)
                        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(df))

}

#[pyfunction]
pub fn rs_levenshtein_dist(s1:&str, s2:&str) -> usize {

    // https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm

    let len1: usize = s1.len();
    let len2: usize = s2.len();
    let mut dp: Vec<Vec<usize>> = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize the first row and first column
    for i in 0..=len1 {
        dp[i][0] = i;
    }

    for j in 0..=len2 {
        dp[0][j] = j;
    }

    // Fill the dp matrix using dynamic programming
    for (i, char1) in s1.chars().enumerate() {
        for (j, char2) in s2.chars().enumerate() {
            if char1 == char2 {
                dp[i + 1][j + 1] = dp[i][j];
            } else {
                dp[i + 1][j + 1] = 1 + dp[i][j].min(dp[i][j + 1].min(dp[i + 1][j]));
            }
        }
    }

    dp[len1][len2]
}

#[pyfunction]
pub fn rs_snowball_stem(word:&str, no_stopwords:bool) -> PyResult<String> {
    if let Some(good) = snowball_stem(word, no_stopwords) {
        Ok(good)
    } else {
        Ok("".to_string())
    }
}

#[pyfunction]
pub fn rs_ref_table(
    pydf: PyDataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc: u32
    , max_feautures: u32
) -> PyResult<PyDataFrame> {
    
    // get_ref_table assumes all docs in df[c] are already lowercased

    let df: DataFrame = pydf.into();
    let out: DataFrame = get_ref_table(df, c,stemmer, min_dfreq, max_dfreq, max_word_per_doc, max_feautures)
                        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(out))
}

#[inline]
pub fn snowball_stem(word:&str, no_stopwords:bool) -> Option<String> {

    if (no_stopwords) & (EN_STOPWORDS.contains(&word)) {
        None
    } else if word.parse::<f64>().is_ok() {
        None
    } else {
        let mut env: SnowballEnv<'_> = SnowballEnv::create(word);
        algorithms::english_stemmer::stem(&mut env);
        Some(env.get_current().to_string())
    }
}

#[inline]
fn snowball_on_series(
    words: Series
) -> Result<Option<Series>, PolarsError> {

    Ok(Some(
        words.utf8()?.par_iter()
        .map(|word| {
            match word {
                Some(w) => Ok(snowball_stem(w, true)),
                _ => Ok(None)
            } 
        }).collect::<PolarsResult<ChunkedArray<Utf8Type>>>()?.into_series()
    ))
}

#[inline]
pub fn get_ref_table(
    df: DataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc: u32
    , max_feautures: u32
) -> PolarsResult<DataFrame> {

    // this function assumes all documents in df[c] are lowercased.
    let stemmer_expr:Expr = match stemmer.to_lowercase().as_str() {
        "snowball" => col(c).map(snowball_on_series, GetOutput::from_type(DataType::Utf8)).alias(&"ref"),
        _ => col(c).alias(&"ref")
    };

    let height: f32 = df.height() as f32;
    let min_count: u32 = (height * min_dfreq).ceil() as u32;
    let max_count: u32 = (height * max_dfreq).ceil() as u32;
    let output: DataFrame = df.select([c])?
    .lazy()
    .with_row_count(&"i", None)
    .select([
        col(&"i")
        , col(c).str().extract_all(lit(r"(?u)\b\w\w+\b")).list().head(lit(max_word_per_doc))
    ]).explode([col(c)])
    .filter(col(c).str().lengths().gt(lit(2)).and(col(c).is_not_null()))
    .select([
        col(&"i")
        , col(c)
        , stemmer_expr // stemmed words, column name is ref
    ]).groupby([col(&"ref")])
    .agg([
        col(c).unique()
        , col(&"i").n_unique().alias(&"doc_cnt")
    ]).filter(
        (col(&"doc_cnt").gt_eq(min_count)).and(col(&"doc_cnt").lt_eq(max_count))
    ).top_k(max_feautures, [col(&"doc_cnt")], [true], true, false)
    .select([
        col(&"ref")
        , (lit("(") + col(c).list().join(&"|") + lit(")")).alias(&"captures")
        , (col(&"doc_cnt").cast(DataType::Float32)/lit(height)).alias(&"doc_freq")
        , (lit(height + 1.)/(col(&"doc_cnt") + lit(1)).cast(DataType::Float64))
            .log(std::f64::consts::E).alias(&"smooth_idf")
    ]).collect()?;
    
    Ok(output)

}

pub fn count_vectorizer(
    df: DataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc: u32
    , max_feautures: u32
    , lowercase: bool
) -> PolarsResult<DataFrame> {

    // lowercase is expensive, do it only once.

    // This, in fact, is still doing some duplicate work.
    // The get_ref_table call technically has computed all count/tfidf vectorizer information
    // at least once.
    // The with_columns(exprs) call technically can be skipped. But the problem is for better 
    // user-experience, we want to use dataframe as output.
    // Scikit-learn uses sparse matrix as output, so technically get_ref_table is all Scikit-learn needs.
    // With that said, this is still faster than Scikit-learn, and does stemming, does more 
    // filtering, and technically computes some stuff twice,

    let mut df_local: DataFrame = df.clone();
    if lowercase {
        df_local = df.lazy().with_column(col(c).str().to_lowercase().alias(c)).collect()?;
    }

    let stemmed_vocab: DataFrame = get_ref_table(
                                            df_local.clone(),
                                            c,
                                            stemmer,
                                            min_dfreq, 
                                            max_dfreq, 
                                            max_word_per_doc, 
                                            max_feautures
                                    )?.sort(["ref"], false, false)?;

    // let mut exprs: Vec<Expr> = Vec::with_capacity(stemmed_vocab.height());
    
    let temp: &Series = stemmed_vocab.column("ref")?;
    let stems: &ChunkedArray<Utf8Type> = temp.utf8()?;
    let temp: &Series = stemmed_vocab.column(&"captures")?;
    let vocabs: &ChunkedArray<Utf8Type> = temp.utf8()?;

    let mut exprs:Vec<Expr> = Vec::with_capacity(stems.len());
    for (stem, pat) in zip(stems.into_iter(), vocabs.into_iter()) {
        if let (Some(s), Some(p)) = (stem, pat) {
            exprs.push(col(c).str().count_match(p).suffix(format!("::cnt_{}", s).as_ref()))
        }
    }

    let out: DataFrame = df_local.lazy().with_columns(exprs).drop_columns([c]).collect()?;
    Ok(out)

}

pub fn tfidf_vectorizer(
    df: DataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc: u32
    , max_feautures: u32
    , lowercase: bool
) -> PolarsResult<DataFrame> {

    // lowercase is expensive, do it only once.

    let mut df_local: DataFrame = df.clone();
    if lowercase {
        df_local = df.lazy().with_column(col(c).str().to_lowercase().alias(c)).collect()?;
    }

    let stemmed_vocab: DataFrame = get_ref_table(
                                            df_local.clone(),
                                            c,
                                            stemmer,
                                            min_dfreq, 
                                            max_dfreq, 
                                            max_word_per_doc, 
                                            max_feautures
                                    )?.sort(["ref"], false, false)?;

    let temp: &Series = stemmed_vocab.column(&"ref")?;
    let stems: &ChunkedArray<Utf8Type> = temp.utf8()?;
    let temp: &Series = stemmed_vocab.column(&"captures")?;
    let vocabs: &ChunkedArray<Utf8Type> = temp.utf8()?;
    let temp: &Series = stemmed_vocab.column(&"smooth_idf")?;
    let smooth_idf: &ChunkedArray<Float64Type> = temp.f64()?;
                                        
    let mut exprs: Vec<Expr> = Vec::with_capacity(stems.len());
    for ((stem, pat), idf) in 
        stems.into_iter().zip(vocabs.into_iter()).zip(smooth_idf.into_iter()) 
    {
        if let (Some(w), Some(p), Some(f)) = (stem, pat, idf) {
            exprs.push(
                (   lit(f).cast(DataType::Float64)
                    * col(c).str().count_match(p).cast(DataType::Float64)
                    / col(&"__doc_len__")
                ).suffix(format!("::tfidf_{}", w).as_ref())
            )
        }
    }
    let out: DataFrame = df_local.lazy().with_column(
        col(c).str().extract_all(lit(r"(?u)\b\w\w+\b"))
        .list().head(lit(max_word_per_doc))
        .list().lengths()
        .cast(DataType::Float64).alias(&"__doc_len__")
    ).with_columns(exprs)
    .drop_columns([c, &"__doc_len__"]).collect()?;
    
    Ok(out)
}