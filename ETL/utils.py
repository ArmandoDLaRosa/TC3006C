import pandas as pd
import pickle
pd.options.mode.chained_assignment = None

class Maps:
    sex = {
        "male"  :   0,
        "female":   1
    }
    embarked = {
        "C" :   0,
        "Q" :   1,
        "S" :   2
    }
    var_types =  {
        "number"    :   "median",
        "object"    :   "mode"
    }
    age_predictors = ["Pclass", "Sex", "Fare"]
    survival_predictors = ['PassengerId', 'Name', 'SibSp', 'Ticket', 'Cabin']
    
def age_imputing(df:pd.DataFrame) -> pd.DataFrame:
    with open('./age_model' , 'rb') as f:
        model = pickle.load(f)
    df_null = df[df["Age"].isnull()]
    not_null =  df[~df["Age"].isnull()]
    df_null["Age"] = model.predict(df_null[Maps.age_predictors])
    df_null["Age"] = df_null["Age"].round()
    return pd.concat([not_null, df_null])
    
def imputing(df:pd.DataFrame) -> pd.DataFrame:
    for var_type in list(Maps.var_types.keys()):
        for col in df.drop(columns = ["Age"]).select_dtypes(include=var_type).columns:
            df[col] = df[col].fillna(getattr(df[col], Maps.var_types[var_type])())
    return df

def encoding(df:pd.DataFrame) -> pd.DataFrame:
    df['Sex'] = df['Sex'].map(Maps.sex)
    df['Embarked'] = df['Embarked'].map(Maps.embarked)
    return df