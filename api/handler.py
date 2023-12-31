import os
import pickle
import pandas as pd
from flask              import Flask, request ,Response
import json
from taxi.Taxi import Taxi 
# load model
# maquina local
#model = pickle.load( open( 'api/model/predict_model.pkl', 'rb') ) 

# produção
model = pickle.load( open( 'model/predict_model.pkl', 'rb') )

# inicialize API
app = Flask(__name__)

@app.route('/taxi/predict', methods = ['POST'])
def taxi_predict():
    test_json = request.get_json()
    test_json = json.loads(test_json)

    if test_json:

        if isinstance(test_json, dict): # Unique Example
            test_raw = pd.DataFrame( test_json, index=[0])
        
        else: # Multiple Example
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
        
        # Instanciar a taxi Class AQUI QUE A MAGIA ACONTECE
        pipeline = Taxi()
        
        df = pipeline.snake_case_columns(df = test_raw)
        
        df1 = pipeline.feature_engeneering(df = df)
        
        df2 = pipeline.preparacao_dados(df = df1)
        
        df3 = pipeline.filter_data(df = df2)
        
        df4 = pipeline.selecao_atributos(df = df3)
        
        df_responde = pipeline.get_prediction(model, test_raw, df4)
        return df_responde
    
    else:
        return Response({}, status=200, minetype='application/json')

if __name__ == '__main__':
    port = int( os.environ.get( 'PORT', 5000 ) )
    app.run('0.0.0.0')