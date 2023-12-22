import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
from typing import List
from math   import radians, cos, sin, asin, sqrt

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def load_dataset(filter: List[int], test: pd.DataFrame) -> str:
    # Filtra o conjunto de dados para incluir apenas as lojas cujos IDs estão presentes na lista fornecida
    test = test[test['TAXI_ID'].isin(filter)]
    
    if not test.empty:
        # Converte o DataFrame resultante em formato JSON
        data = json.dumps(test.to_dict(orient='records'))
    else:
        data = 'error'  

    return data

def get_predictions(data: str) -> pd.DataFrame:
    #url = 'http://localhost:5000/taxi/predict'
    url = 'https://taxi-4ui9.onrender.com/taxi/predict'
    
    headers = {'Content-type': 'application/json'}
    try:
        r = requests.post(url, json=data, headers=headers)
        r.raise_for_status()  # Raise an exception for HTTP errors (non-2xx responses)
        
        df_result = pd.DataFrame(r.json(), columns= r.json()[0].keys())
        return df_result
    
    except requests.exceptions.RequestException as e:
        st.write(f"Error occurred during prediction: {e}")
        return pd.DataFrame()
    
def haversine(lon1, lat1, lon2, lat2):
    
    # converter decimal para radiano
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = (np.sin(dlat/2)**2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2)**2)
    r = 6371 # Raio da Terra em quilômetros. Use 3956 para milhas.
    d = 2 * r * np.arcsin(np.sqrt(a))
    
    return np.round(d,5)