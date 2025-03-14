# Car Price Prediction Site

A web application built with FastAPI to predict car prices based on machine learning models. The app allows users to input car characteristics and get a predicted price, all packaged in a Docker container for easy deployment.

## Features
- **Interactive Web Form**: Input car details like brand, model, milige, and more to get a price prediction.
- **Machine Learning**: Uses trained model LightGBM and pipeline (`model.pkl` and `pipeline.pkl`) for accurate predictions.
- **Docker Support**: Easily deployable with a single Docker command.
- **Real-Time Validation**: Client-side validation for fields like milige and model year.
- **Autocomplete**: Searchable dropdowns for brands, models, and colors.

## Guide to run progect
Using docker:
- git clone https://github.com/ITProger123/car-price-prediction-site.git
- cd car-price-prediction-site
- docker build -t car-price-prediction .
- docker run -d -p 1250:8000 car-price-predictor

Without docker:
- git clone https://github.com/ITProger123/car-price-prediction-site.git
- cd car-price-prediction-site
- pip install -r requirements.txt
- uvicorn main:app --host 0.0.0.0 --port 8000
