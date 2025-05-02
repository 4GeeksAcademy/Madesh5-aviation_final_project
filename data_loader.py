import pandas as pd
import joblib
from config import MODEL_PATH, DATA_PATH

def load_data():
    return pd.read_csv(DATA_PATH)

def load_model():
    return joblib.load(MODEL_PATH)