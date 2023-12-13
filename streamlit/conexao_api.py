import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
from typing import List

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def load_dataset(store_ids: List[int], test: pd.DataFrame, store: pd.DataFrame) -> str:

    # Realiza o merge do conjunto de dados de teste com as informações das lojas
    df_test = pd.merge(test, store, how='left', on='Store')

    # Filtra o conjunto de dados para incluir apenas as lojas cujos IDs estão presentes na lista fornecida
    df_test = df_test[df_test['Store'].isin(store_ids)]

    if not df_test.empty:
        # Remove os dias em que as lojas estavam fechadas ('Open' == 0) e as linhas com valores nulos na coluna 'Open'
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]

        if 'Id' in df_test.columns:
            # Remove a coluna 'Id' do conjunto de dados
            df_test = df_test.drop('Id', axis=1)
        else:
            pass
        
        # Converte o DataFrame resultante em formato JSON
        data = json.dumps(df_test.to_dict(orient='records'))
    else:
        data = 'error'

    return data

def get_predictions(data: str) -> pd.DataFrame:

    url = 'http://localhost:5000/taxi/predict'
    #url = 'https://rossman.onrender.com/taxi/predict'
    
    headers = {'Content-type': 'application/json'}
    try:
        r = requests.post(url, json=data, headers=headers)
        r.raise_for_status()  # Raise an exception for HTTP errors (non-2xx responses)
        
        df_result = pd.DataFrame(r.json(), columns= r.json()[0].keys())
        return df_result
    
    except requests.exceptions.RequestException as e:
        st.write(f"Error occurred during prediction: {e}")
        return pd.DataFrame(columns=['store', 'prediction'])