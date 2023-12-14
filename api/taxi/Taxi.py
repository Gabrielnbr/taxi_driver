import math
import pickle
import numpy                as np
import pandas               as pd
import datetime
import inflection


class Taxi (object):
    def __init__ (self):
        self.home_path = ''
        #self.st = pickle.load(open(self.home_path + '../feature/haversine_km_ss.pkl', 'rb'))
    
    # Padroniza as nomeclatura das colunas para snake_case
    def snake_case_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        # Transforma todas para Snakecase
        snakecase = lambda x : inflection.underscore( x )
        colunas = df_copy.columns.to_list()

        # Mapeia as colunas e aplica o snakecase no nome das colunas
        cols_new = list( map( snakecase, colunas) )

        #renomeia as colunas
        df_copy.columns = cols_new

        return df_copy
    
    # Métrica de avaliação definida pelo negócio.
    # def haversine(lon1, lat1, lon2, lat2):
    
    #     # converter decimal para radiano
    #     lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    #     # haversine formula 
    #     dlon = lon2 - lon1 
    #     dlat = lat2 - lat1 
    #     a = (np.sin(dlat/2)**2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2)**2)
    #     r = 6371 # Raio da Terra em quilômetros. Use 3956 para milhas.
    #     d = 2 * r * np.arcsin(np.sqrt(a))
    #     return np.round(d,5)
    
    # Feature_engeneering
    def feature_engeneering(self, df: pd.DataFrame) -> pd.DataFrame:
        # 2 snake_case ==========================================
        # df = Taxi.snake_case_columns(df)
        
        # 3.1 valores_NAN ================================================
        df['origin_call'] = np.where(df['call_type'] == 'A', 1, 0)
        df['origin_stand'] = np.where(df['call_type'] == 'B', 1, 0)

        # 3.2 Modificação timestamp =====================================
        df['timestamp'] = df['timestamp'].astype(float)

        df["data_timestamp"] = np.where(df["timestamp"].notnull(), 
                                                pd.to_datetime(df["timestamp"], unit='s',utc=True),
                                                np.nan)

        # Derivação da data
        df["year"] = df["data_timestamp"].dt.year
        df["month"] = df["data_timestamp"].dt.month
        df["day"] = df["data_timestamp"].dt.day
        df["hour"] = df["data_timestamp"].dt.hour
        df["min"] = df["data_timestamp"].dt.minute
        df["weekday"] = df["data_timestamp"].dt.weekday
        df['semana_do_ano'] = df['data_timestamp'].dt.isocalendar().week
        df['ano_semana'] = df['data_timestamp'].dt.strftime('%Y-%W')

        # 3.3 Lat Long ===============================================================
        df["lat_long_temp"] = np.where(df["polyline"] == '[]', 0, 
                                            df["polyline"].str.replace(r"[[|[|]|]|]]", "").str.split(",")
                                            )

        df["long_inicial"] = df["lat_long_temp"].apply(lambda x: float(x[0]) if x else 0)
        df["lat_inicial"] = df["lat_long_temp"].apply(lambda x: float(x[1]) if x else 0)
        df["long_final"] = df["lat_long_temp"].apply(lambda x: float(x[-2]) if x else 0)
        df["lat_final"] = df["lat_long_temp"].apply(lambda x: float(x[-1]) if x else 0)

        df['delta_lat'] = df["lat_final"] - df["lat_inicial"]
        df['delta_long'] = df["long_final"] - df["long_inicial"]

        # 3.4 missing data =================================================================
        df["missing_data"] = np.where(df["missing_data"] == False, 0, 1)

        # 3.5 day_type =============================================
        df.drop("day_type", axis=1, inplace=True)

        # 3.6 drop_columns ==============================================
        df.drop(columns = ['timestamp','polyline', 'data_timestamp', 'lat_long_temp'], axis = 1, inplace=True)

        # 3.7 haversine ================================================
        #df['haversine_km'] = haversine(df["long_inicial"], df["lat_inicial"], df["long_final"], df["lat_final"])
        #df['haversine_mt'] = df['haversine_km']*1000

        return df

    # Filter Data
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:

        # 5. Filtro do Dataset
        filtro_lat = ((df['lat_inicial'] >= 38.4) & (df['lat_inicial'] <= 42.2) &
                      (df['lat_final'] >= 38.4) & (df['lat_final'] <= 42.2))

        filtro_long = ((df['long_inicial'] >= -9.5) & (df['long_inicial'] <= -6.0) &
                       (df['long_final'] >= -9.5) & (df['long_final'] <= -6.0))

        df_filter = df.loc[filtro_lat & filtro_long]

        df_filter = df_filter[((df_filter['delta_long'] <= 0.2) & (df_filter['delta_long'] >= -0.2) &
                                (df_filter['delta_lat'] <= 0.2) & (df_filter['delta_lat'] >= -0.2))]
        return df_filter

    # Preparação dos dados
    def preparacao_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        # 6.1 Encoding =============================================
        df = pd.get_dummies(df, columns=['call_type'])

        # 6.2. Rescaling ===============================================
        #st = pickle.load(open('../src/features/haversine_km_ss.pkl', 'rb'))
        #df['haversine_km'] = st.fit_transform(df[['haversine_km']])

        df['delta_lat'] = np.log1p(df['delta_lat'].values)
        df['delta_long'] = np.log1p(df['delta_long'].values)

        # 6.3. Variáveis Cíclicas ======================================

        df['month_sin'] = df['month'].apply( lambda x: np.sin( x * (2 * np.pi/12 ) ) )
        df['month_cos'] = df['month'].apply( lambda x: np.cos( x * (2 * np.pi/12 ) ) )

        # day
        df['day_sin'] = df['day'].apply( lambda x: np.sin( x * (2 * np.pi/30 ) ) )
        df['day_cos'] = df['day'].apply( lambda x: np.cos( x * (2 * np.pi/30 ) ) )

        # week of year
        df['semana_do_ano_sin'] = df['semana_do_ano'].apply( lambda x: np.sin( x * (2 * np.pi/52 ) ) )
        df['semana_do_ano_cos'] = df['semana_do_ano'].apply( lambda x: np.cos( x * (2 * np.pi/52 ) ) )

        # day of week
        df['weekday_sin'] = df['weekday'].apply( lambda x: np.sin( x * (2 * np.pi/7 ) ) )
        df['weekday_cos'] = df['weekday'].apply( lambda x: np.cos( x * (2 * np.pi/7 ) ) )

        return df
    
    # Selação de Atributos
    def selecao_atributos(self, df: pd.DataFrame) -> pd.DataFrame:

        # 7.0 Seleção dos atributos ==========================
        cols_drop =(['year', 'month', 'day', 'semana_do_ano', 'weekday', 'ano_semana', 
                     #'haversine_mt', 
                     'semana_do_ano', 'trip_id', 'taxi_id'])
        df = df.drop(cols_drop, axis=1)
        df = df.drop(['long_final', 'lat_final'], axis=1)
        df = (df[['delta_lat', 'lat_inicial', 'long_inicial',
                  #'haversine_km',
                  'delta_long']])

        return df
    
    def get_prediction(self, model, original_data, test_data, processed_data):
        # prediction
        pred = model.predict(test_data)
        
        original_data['predicted_lat'] = pred.T[1]
        original_data['predicted_long'] = pred.T[0]
        
        return original_data.to_json( orient = 'records', date_format = 'iso' )