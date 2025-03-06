import os
import sys
sys.path.append(os.getcwd())
import uuid

from functools import lru_cache
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from models import Auto
import pandas as pd
from ML.preprocess import extract_fetures_from_engine, preprocess_transmission, extract_speed, compress_similar_gradation
from ML.preprocess import add_is_luxuary_model, add_is_luxuary_brand, preprocess, clean_columns
import joblib
from typing import Annotated


# model = joblib.load('model.pkl')
# pipeline = joblib.load('pipeline.pkl')
templates = Jinja2Templates(directory='.')

app = FastAPI()


def get_ml_models():
    model = joblib.load('model.pkl')
    pipeline = joblib.load('pipeline.pkl')
    return model, pipeline

df = pd.read_csv('data/train.csv')

@lru_cache
def get_brands():
    return sorted(list(df['brand'].unique()))


@lru_cache
def get_models():
    return sorted(list(df['model'].unique()))

@lru_cache
def get_ext_col():
    return sorted(list(df['ext_col'].str.capitalize().unique()))

@lru_cache
def get_int_col():
    return sorted(list(df['int_col'].unique()))


@app.get("/", response_class=HTMLResponse)
def get_form(request: Request, brands: list = Depends(get_brands), models: list = Depends(get_models), ext_cols: list = Depends(get_ext_col), int_cols: list = Depends(get_int_col)):
    return templates.TemplateResponse("app/index.html", {'request': request, 'brands': brands, 'models': models, 'ext_cols': ext_cols, 'int_cols': int_cols})



@app.post('/predict_price')
def input_data(auto: Annotated[Auto, Form()]):
    try:
        model, pipeline = get_ml_models() 
        auto_dict = auto.model_dump()
        auto_id = pd.DataFrame([uuid.uuid4()], columns=['id'])
        auto_df = pd.concat([auto_id, pd.DataFrame([auto_dict])], axis=1)
        transform_auto = pipeline.transform(auto_df)
        predict_price = np.exp(model.predict(transform_auto)[0])
        return {'predicted_price': predict_price}
    except Exception as e:
        return JSONResponse(status_code=400, content={'error': str(e)})




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)