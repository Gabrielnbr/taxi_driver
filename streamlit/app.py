import pandas as pd
import streamlit as st

from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster

import json
import folium

from conexao_api import load_dataset, get_predictions, haversine

def pg_inicial():
    st.title('Seja bem vindo ao projeto Taxi Drive de Previsão de mobilidade urbana')
    
    orientacao = '''    <p>Caso queira acessar mais informações específicas do projeto, segue os links a baixo:</p>
                    <p><i class="fa-brands fa-linkedin"></i><a href="https://www.linkedin.com/in/gabriel-nobre-galvao/"> Linkedin: Aprendizado do projeto Taxi Drive de mobilidade urbana</a></p>
                    <p><i class="fa-brands fa-medium"></i><a href="https://medium.com/@gabrielnobregalvao/taxi-drive-projeto-de-previs%C3%A3o-de-mobilidade-urbana-d921f895f9af"> Medium: Taxi Drive: Projeto de Previsão de mobilidade urbana</a></p>
                    <p><i class="fa-brands fa-github"></i><a href="https://github.com/Gabrielnbr/taxi_driver"> Git Hub: Taxi Drive</a></p>
                    '''
    
    st.write(orientacao, unsafe_allow_html = True)
    
    st.subheader("Contato Profissional")
    
    contato = '''
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
    
    <ul class="actions">
        <table>
            <tr>
                <th><i class="fa-solid fa-folder-tree"></i><a href="https://bit.ly/portfolio-gabriel-nobre"> Portfólio de Projetos</a></th>
                <th><i class="fa-brands fa-linkedin"></i><a href="https://www.linkedin.com/in/gabriel-nobre-galvao/"> Linkedin</a></th>
                <th><i class="fa-brands fa-medium"></i><a href="https://medium.com/@gabrielnobregalvao"> Medium</a></th>
                <th><i class="fa-brands fa-github"></i><a href="https://github.com/Gabrielnbr"> Git Hub</a></th>
                <th><i class="fa-solid fa-envelope"></i><a href="mailto:gabrielnobregalvao@gmail.com"> E-mail</a></th>
            </tr>
        </table>
    </ul>
    '''
    st.write(contato, unsafe_allow_html=True) 

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
    
    info , predicao = st.tabs(["Informações", "Predição"])
    
    with info:
        pg_inicial()
    
    with predicao:
        app(test=test)
