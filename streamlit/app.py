import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly_express as px

import json

from matplotlib import pyplot as plt

from conexao_api import load_dataset, get_predictions

def app(test):
    
    
    index = [0,5,10,20,319]
    
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
    
    problema ='''Traceback (most recent call last):
    
              File "D:/Git_hub/venv310/lib/site-packages/flask/app.py", line 2077, in wsgi_app
                response = self.full_dispatch_request()
                
              File "D:/Git_hub/venv310/lib/site-packages/flask/app.py", line 1525, in full_dispatch_request
                rv = self.handle_user_exception(e)
                
              File "D:/Git_hub/venv310/lib/site-packages/flask/app.py", line 1523, in full_dispatch_request
                rv = self.dispatch_request()
                
              File "D:/Git_hub/venv310/lib/site-packages/flask/app.py", line 1509, in dispatch_request
                return self.ensure_sync(self.view_functions[rule.endpoint])(**req.view_args)
                
              File "D:/Git_Hub/Seleções de Trabalho/2023.2 - Aceleração de Carreira/api/handler.py", line 24, in taxi_predict       
                test_raw = pd.DataFrame( json.loads(test_json), columns = test_json[0].keys())
                
              File "C:/Users/gbnob/AppData/Local/Programs/Python/Python310/lib/json/__init__.py", line 339, in loads
                raise TypeError(f'the JSON object must be str, bytes or bytearray, '
                
            TypeError: the JSON object must be str, bytes or bytearray, not list'''
    
    st.write(problema)
    
    problema1 = ''''[2023-12-13 05:08:02,998] ERROR in app: Exception on /taxi/predict [POST]
    
                        Traceback (most recent call last):
                          File "D:/Git_hub/venv310/lib/site-packages/flask/app.py", line 2077, in wsgi_app
                            response = self.full_dispatch_request()
                            
                          File "D:/Git_hub/venv310/lib/site-packages/flask/app.py", line 1525, in full_dispatch_request
                            rv = self.handle_user_exception(e)
                            
                          File "D:/Git_hub/venv310/lib/site-packages/flask/app.py", line 1523, in full_dispatch_request
                            rv = self.dispatch_request()
                            
                          File "D:/Git_hub/venv310/lib/site-packages/flask/app.py", line 1509, in dispatch_request
                            return self.ensure_sync(self.view_functions[rule.endpoint])(**req.view_args)
                            
                          File "D:/Git_Hub/Seleções de Trabalho/2023.2 - Aceleração de Carreira/api/handler.py", line 24, in taxi_predict 
                                
                        AttributeError: 'str' object has no attribute 'keys'

                        '''
    st.write(problema1)
    
    st.header("GET_PREDICTION")
    df = get_predictions(df_test)
    st.dataframe(df)

if __name__ == "__main__":
    
    test = pd.read_csv("streamlit/data/test.csv")
    
    app(test=test)