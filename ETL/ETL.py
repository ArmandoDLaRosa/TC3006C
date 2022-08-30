import utils
import pandas as pd

def extraction(path:str)  -> pd.DataFrame:
    return pd.read_csv(path)

def transform(df:pd.DataFrame) -> pd.DataFrame:
    # Feature Selection
    df = df.drop(columns=utils.Maps.survival_predictors)

    # Feature Engineering
        # Jobs - Mrs, Miss ************* MISsING considerarlo en la age
        # Sib Ranges
        # Age Ranges
    
    # Data Enriching

    # Encoding
    df = utils.encoding(df)        
    # Missing Values  - Hard fill
    df = utils.imputing(df)    
    # Missing Values -  smart fill
    df = utils.age_imputing(df)
    return df

def load(df:pd.DataFrame, path:str) -> pd.DataFrame:
    df.to_csv(path)
    return df

def etl(extract_path:str="Titanic/train.csv", load_path:str="./output", return_df:bool = True) -> pd.DataFrame:
    return load(transform(extraction(extract_path)), load_path) if return_df else "dataset output -done"

if __name__ == '__main__':
    etl()