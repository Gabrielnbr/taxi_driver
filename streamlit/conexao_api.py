import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
from typing import List

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def load_dataset(ids: List[int], test: pd.DataFrame) -> str:
    # Filtra o conjunto de dados para incluir apenas as lojas cujos IDs estão presentes na lista fornecida
    test = test[test.index.isin(ids)]
    
    if not test.empty:
        # Converte o DataFrame resultante em formato JSON
        data = json.dumps(test.to_dict(orient='records'))
    else:
        data = 'error'

    return data

def get_predictions(data: str) -> pd.DataFrame:
    url = 'http://localhost:5000/taxi/predict'
    #url = 'https://taxi-4ui9.onrender.com/taxi/predict'
    
    headers = {'Content-type': 'application/json'}
    try:
        r = requests.post(url, json=data, headers=headers)
        r.raise_for_status()  # Raise an exception for HTTP errors (non-2xx responses)
        
        df_result = pd.DataFrame(r.json(), columns= r.json()[0].keys())
        return df_result
    
    except requests.exceptions.RequestException as e:
        st.write(f"Error occurred during prediction: {e}")
        return pd.DataFrame(columns=['store', 'prediction'])