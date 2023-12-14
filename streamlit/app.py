import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly_express as px

import json

from matplotlib import pyplot as plt

from conexao_api import load_dataset, get_predictions

def app(test):
    
    
    index = [0]
    
    st.header("TEST_RAW")
    st.dataframe(test)
    
    st.header("TEST_FILTER_INDEX")
    test_2 = test[test.index.isin(index)]
    st.dataframe(test_2)
    
    st.header("LOAD_DATA_SET_JSON")
    df_test = load_dataset(index,test)
    st.write(df_test)
    
    st.header("LOAD_DATA_SET_QUANDO_A_API_RECEBER")
    test_json = json.loads(df_test)
    st.write(test_json)
    
    
    
    st.dataframe(pd.DataFrame(test_json, columns = test_json[0].keys()))
    
    
    
    st.header("GET_PREDICTION")
    df = get_predictions(df_test)
    st.dataframe(df)

if __name__ == "__main__":
    
    test = pd.read_csv("streamlit/data/test.csv")
    
    app(test=test)