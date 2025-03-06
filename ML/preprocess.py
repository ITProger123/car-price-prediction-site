import pandas as pd
import re
import numpy as np
from category_encoders.binary import BinaryEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import datetime
import joblib

train_df = pd.read_csv("data/train.csv")
columns = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'model_year', 'milage']
def extract_fetures_from_engine(engine_str: str) -> tuple:
    hp_match = re.search(r'(\d+\.?\d*)HP', engine_str)  
    engine_size_match = re.search(r'(\d+\.?\d*)L', engine_str) 
    cylinders_match = re.search(r'(\d+)\sCylinder', engine_str)
    hp = float(hp_match.group(1)) if hp_match else None
    engine_size = float(engine_size_match.group(1)) if engine_size_match else None
    cylinders = int(cylinders_match.group(1)) if cylinders_match else None
    return hp, engine_size, cylinders

def preprocess_transmission(x: str) -> str:
    type_arrays = x.split()
    if x == 'Transmission w/Dual Shift Mode':
        return x
    elif 'CVT' in type_arrays:
        return 'CVT'
    elif 'A/T' in type_arrays or 'AT' in type_arrays or 'Automatic' in type_arrays:
        return 'A/T'
    elif 'M/T' in type_arrays:
        return 'M/T'
    return 'Other'


def extract_speed(x: str) -> int:
    return int(match.group(1)) if (match := re.search(r'(\d+)-Speed', x)) else None


def compress_similar_gradation(df: pd.DataFrame) -> pd.DataFrame:
    df['fuel_type'] = df['fuel_type'].replace(['–', 'not supported'], 'Unknown')
    df['transmission'] = df['transmission'].replace(['–', '2', 'SCHEDULED FOR OR IN PRODUCTION'], 'Unknown')
    df['transmission'] = df['transmission'].replace('Automatic', 'A/T')
    df['transmission'] = df['transmission'].replace('Manual', 'M/T')
    df['transmission'] = df['transmission'].apply(lambda x: re.sub(r'(\d+)-Speed Automatic', r'\1-Speed A/T', x))
    df['transmission'] = df['transmission'].apply(lambda x: re.sub(r'(\d+)-Speed Manual', r'\1-Speed M/T', x))
    df['transmission'] = df['transmission'].apply(lambda x: re.sub(r'CVT.*', 'CVT', x))
    df['int_col'] = df['int_col'].str.capitalize()
    df['ext_col'] = df['ext_col'].str.capitalize()
    return df


luxuary_brands = train_df.groupby("brand")["price"].mean().sort_values(ascending=False)[:10].index
luxuary_models = train_df.groupby("model")["price"].mean().sort_values(ascending=False)[:10].index

def add_is_luxuary_model(df: pd.DataFrame, models: list) -> pd.DataFrame:
    df['is_luxuary_model'] = df['model'].apply(lambda x: 1 if x in models else 0)
    return df


def add_is_luxuary_brand(df: pd.DataFrame, brands: list) -> pd.DataFrame:
    df['is_luxuary_brand'] = df['brand'].apply(lambda x: 1 if x in brands else 0)
    return df



def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = add_is_luxuary_model(df, luxuary_models)
    df = add_is_luxuary_brand(df, luxuary_brands)
    df = compress_similar_gradation(df)
    df['num_speeds'] = df['transmission'].apply(extract_speed)
    df['transmission'] = df['transmission'].apply(preprocess_transmission)
    df[['hp', 'engine_size', 'cylindres']] = df['engine'].apply(lambda x: pd.Series(extract_fetures_from_engine(x)))
    df['auto_age'] = df['model_year'].apply(lambda x: datetime.datetime.now().year - x)
    df.drop(columns=['engine', 'model_year', 'id', 'clean_title'], inplace=True)
    return df

def clean_columns(df: pd.DataFrame):
    df_columns = df.columns
    func = lambda x: re.sub(r"^[^_]+_+", "", x)
    new_columns = list(map(func, df_columns))
    df.rename(columns=dict(zip(df_columns, new_columns)), inplace=True)
    return df



columns_to_target_encoders = ["model", "brand"]
columns_to_one_hot_encoder = ["fuel_type", "transmission", "accident"]
columns_to_binary_encoder = ["int_col", "ext_col"]

cat_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col',
       'int_col', 'accident']
num_cols = ['model_year', 'milage']


columns_after = ['num_speeds', 'hp', 'engine_size', 'cylindres', 'milage', 'auto_age']




fill_missing = ColumnTransformer(
    transformers=[('num_', SimpleImputer(strategy='most_frequent'), cat_cols),
                  ('cat_', SimpleImputer(strategy='median'), num_cols)],
                  remainder='passthrough'
).set_output(transform="pandas")

fill_pipeline = Pipeline(
    [
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())  
    ]
)


fill_missing_after = ColumnTransformer(transformers=[('fill_', fill_pipeline, columns_after),
                                                     
                                                     ],
                                                     remainder='passthrough').set_output(transform="pandas")



target_pipeline = Pipeline(
    [('target_encoder', TargetEncoder())]
)

one_hot_pipeline = Pipeline(
    [('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))]
)

binary_pipeline = Pipeline(
    [('binary', BinaryEncoder(handle_unknown='ignore'))]
)


encoders = ColumnTransformer(transformers=[
    ('target', target_pipeline, columns_to_target_encoders),
    ('one_hot', one_hot_pipeline, columns_to_one_hot_encoder),
    ('binary', binary_pipeline, columns_to_binary_encoder)
], remainder='passthrough')

pipeline = Pipeline(
    [('fill_missing', fill_missing),
     ('transform1', FunctionTransformer(clean_columns, validate=False)),
    ('feature_engeneering', FunctionTransformer(preprocess, validate=False)),
    ('fill_miss_value', fill_missing_after),
    ('transform2', FunctionTransformer(clean_columns, validate=False)),
    ('encoders', encoders)
     ]
)

X = train_df.drop(columns=['price'])
y = np.log(train_df['price'])


pipeline.fit(X, y)


joblib.dump(pipeline, 'pipeline.pkl')


