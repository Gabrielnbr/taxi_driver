import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly_express as px

import json

from matplotlib import pyplot as plt

from conexao_api import load_dataset, get_predictions

def app(test):
    st.header("Oi Stramlit")
    
    index = [0,5,10,20,319]
    
    
    df_test = load_dataset(index,test)
    st.write(df_test)
    
    test_json = json.loads(df_test)
    
    df = get_predictions(df_test)
    st.dataframe(df)

if __name__ == "__main__":
    
    test = pd.read_csv("streamlit/data/test.csv")
    
    app(test=test)