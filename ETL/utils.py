import pandas as pd
import pickle
pd.options.mode.chained_assignment = None

class MLmodel:
    def __init__(self, name="", accuracy=0, precision=0, recall=0, f1=0):
        self.name = name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
    def to_dict(self):
        return {
            'name': self.name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1     
        }
        
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
    title = {
        # Royal - 0
        "the Countess" : 0,
        "Jonkheer" : 0,
        "Lady" : 0,
        # Normal - 1
        "Don" : 1,
        "Dona" : 1,
        "Miss" : 1,
        "Ms" : 1,
        "Mlle" : 1,
        "Mrs" : 1,
        "Mr" : 1,
        "Sir" : 1,
        "Master" : 1, # Kid
        # Profession - 2
        "Rev" : 2,
        "Capt" : 2,
        "Col" : 2,
        "Dr" : 2,
        "Major" : 2
    }  

    var_types =  {
        "number"    :   "median",
        "object"    :   "mode"
    }
    age_predictors = ["Title", "Sex", "Pclass"] # This isn't used to drop columns
    survival_predictors = ['PassengerId','SibSp', 'Ticket', 'Cabin', 'Parch', 'Fare', 'Name']  # This is used to drops columns
    kaggle_predictors = ['SibSp', 'Ticket', 'Cabin', 'Parch', 'Fare', 'Name']  # This is used to drops columns

def age_imputing(df:pd.DataFrame) -> pd.DataFrame:
    with open('./age_model' , 'rb') as f:
        model = pickle.load(f)
    df_null = df[df["Age"].isnull()]
    not_null =  df[(~df["Age"].isnull()) |  (df["Age"] <= 0)]
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
    df['Title'] = df['Title'].map(Maps.title)
    return df

def title_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["Title"]=[i.split(".")[0].split(",")[-1].strip() for i in df["Name"]]
    return df
