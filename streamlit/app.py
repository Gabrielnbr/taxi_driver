import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly_express as px

from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster

import json
import folium

from matplotlib import pyplot as plt

from conexao_api import load_dataset, get_predictions, haversine

def app(test):
    
    filter = st.multiselect("Selecione o Taxi", test['TAXI_ID'].sort_values().unique(),[20000108])
    teste = test[test['TAXI_ID'].isin(filter)]
    st.dataframe(teste,  use_container_width=True)
    
    if st.button('Predict '):
        
        st.header("GET_PREDICTION")
        df = get_predictions(load_dataset(filter,test))
        
        df['distancia_real_predita'] = haversine(df["long_final"],df["lat_final"],df['predicted_long'],df['predicted_lat'])*1000
        
        st.dataframe(df, use_container_width=True)
        map = mapa(df)
        folium_static(map)

def mapa(df):
    mapa = folium.Map(location=[df['predicted_lat'].median(),df['predicted_long'].median()],
                  zoom_start=12)

    for i, r in df.iterrows():
        
        # LAT E LONG PREDITO
        folium.CircleMarker(location=[df["predicted_lat"][i],df["predicted_long"][i]],
                            radius=0.2,
                            color="blue",
                            popup= 'predicted_lat: {}, predicted_long: {}'.format(df['predicted_lat'][i],
                                                                        df['predicted_long'][i])).add_to(mapa)

        # Isso facilita a visão do atendente para identificar o próximo cliente mais próximo daquele táxi.

        # Estou adicionando um raio de 300 metros ao redor do ponto.
        folium.Circle(
        location=[df["predicted_lat"][i],df["predicted_long"][i]],
        radius=300,
        color="black",
        weight=0.1,
        fill_opacity=0.3,
        opacity=0.1,
        fill_color="yellow",
        fill=False,
        popup="Taxi_ID {}".format(df['TAXI_ID'][i]),
        tooltip="Previsão de Chegada do Taxi da corrida {}".format(df['TRIP_ID'][i])
        ).add_to(mapa)


        # LAT E LONG INICIAL
        folium.CircleMarker(location=[df["lat_inicial"][i],df["long_inicial"][i]],
                            radius=0.2,
                            color="blue",
                            popup= 'lat_inicial: {}, long_inicial: {}'.format(df['lat_inicial'][i],
                                                                        df['long_inicial'][i])).add_to(mapa)
        
        folium.Circle(
        location=[df["lat_inicial"][i],df["long_inicial"][i]],
        radius=300,
        color="black",
        weight=0.5,
        fill_opacity=0.3,
        opacity=0.5,
        fill_color="green",
        fill=False,
        popup="Taxi_ID {}".format(df['TAXI_ID'][i]),
        tooltip="Partida do Taxi da corrida {}".format(df['TRIP_ID'][i])
        ).add_to(mapa)
        
        
        # LAT E LONG FINAL REAL
                # Estou adicionando um raio de 200 metros ao redor do ponto.
        folium.CircleMarker(location=[df["lat_final"][i],df["long_final"][i]],
                            radius=0.2,
                            color="blue",
                            popup= 'lat_final: {}, long_final: {}'.format(df['lat_final'][i],
                                                                        df['long_final'][i])).add_to(mapa)
        
        # Estou adicionando um raio de 100 metros ao redor do ponto.
        folium.Circle(
        location=[df["lat_final"][i],df["long_final"][i]],
        radius=300,
        color="black",
        weight=0.5,
        fill_opacity=0.3,
        opacity=0.5,
        fill_color="red",
        fill=False,
        popup="Taxi_ID {}".format(df['TAXI_ID'][i]),
        tooltip="Chegada real do Taxi da corrida {}".format(df['TRIP_ID'][i])
        ).add_to(mapa)

    return mapa

if __name__ == "__main__":
    
    test = pd.read_csv("streamlit/data/test.csv")
    
    app_st , pg_test_st = st.tabs(["Negócio", "Teste"])
    
    with app_st:
        app(test=test)
    
    #with pg_test_st:
    #    pagina_test(test=test)
    
    




def pagina_test(test):
    
    index = [0,318,319]
    
    st.header("TEST_RAW")
    st.dataframe(test)
    
    st.header("TEST_FILTER_INDEX")
    test_2 = test[test.index.isin(index)]
    st.dataframe(test_2)
    
    st.header("LOAD_DATA_SET_JSON")
    df_test = load_dataset(index,test)
    st.write(df_test)
    
    # ÁREA DE TESTE, NÃO ENVIAR PARA GET_PERDICTION ===================================
    #st.header("LOAD_DATA_SET_QUANDO_A_API_RECEBER")
    #test_json = json.loads(df_test)
    #st.write(test_json)
    #st.dataframe(pd.DataFrame(test_json, columns = test_json[0].keys()))
    # ÁREA DE TESTE, NÃO ENVIAR PARA GET_PERDICTION ===================================
    
    if st.button('Predict'):
        st.header("GET_PREDICTION")
        df = get_predictions(df_test)
        st.dataframe(df)
        
        df['distancia_real_predita'] = haversine(df["long_final"],df["lat_final"],df['predicted_long'],df['predicted_lat'])*1000
        
        map = mapa(df)
        folium_static(map)
