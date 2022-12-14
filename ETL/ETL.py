import utils
import pandas as pd

def extraction(path:str)  -> pd.DataFrame:
    return pd.read_csv(path)

def transform(df:pd.DataFrame, kaggle:bool = False) -> pd.DataFrame:

    # Feature Engineering ---- Jobs 
    df = utils.title_engineering(df)        
    #  Feature Engineering ---- Sib Ranges
    #  Feature Engineering ---- Age Ranges

    # Feature Selection
    df = df.drop(columns=utils.Maps.kaggle_predictors) if kaggle else df.drop(columns=utils.Maps.survival_predictors)

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

def etl(extract_path:str="Titanic/train.csv", load_path:str="./output", return_df:bool = True, kaggle:bool = False) -> pd.DataFrame:
    return load(transform(extraction(extract_path), kaggle), load_path) if return_df else "dataset output -done"

if __name__ == '__main__':
    etl()