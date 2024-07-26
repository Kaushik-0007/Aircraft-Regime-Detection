import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import joblib

data=pd.read_csv('ADSB.csv')
data['alt']=pd.to_numeric(data['alt'],errors='coerce')
data['spd']=pd.to_numeric(data['spd'],errors='coerce')
data['roc']=pd.to_numeric(data['roc'],errors='coerce')
data.dropna(subset=['alt','spd','roc'],inplace=True)

scaler=StandardScaler()
data_scaled=scaler.fit_transform(data[['alt', 'spd', 'roc']])

input_dim=data_scaled.shape[1]
encoding_dim=2

input_layer=Input(shape=(input_dim,))
encoded=Dense(encoding_dim,activation='relu')(input_layer)
decoded=Dense(input_dim,activation='sigmoid')(encoded)

autoencoder=Model(inputs=input_layer, outputs=decoded)
encoder=Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001),loss='mse')

autoencoder.fit(data_scaled,data_scaled,epochs=15,batch_size=32,shuffle=True,verbose=1)

encoded_data=encoder.predict(data_scaled)

n_clusters_list=[4,5,6,7,8]
gmm_models={}

for n_clusters in n_clusters_list:
    gmm=GaussianMixture(n_components=n_clusters,random_state=0)
    clusters=gmm.fit_predict(encoded_data)
    data[f'phase_K{n_clusters}']=clusters
    gmm_models[n_clusters]=gmm
    joblib.dump(gmm, f'gmm_model_{n_clusters}.pkl')

autoencoder.save('autoencoder_model.h5')
encoder.save('encoder_model.h5')
joblib.dump(scaler,'scaler.pkl')
data.to_csv('processed_data.csv',index=False)
