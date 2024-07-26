from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

app=Flask(__name__)

encoder=load_model('encoder_model.h5')
scaler=joblib.load('scaler.pkl')

data=pd.read_csv('processed_data.csv')

k_values=[4,5,6,7,8]
kmeans_models={k: joblib.load(f'kmeans_{k}.pkl') for k in k_values}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    alt=int(request.form['alt'])
    spd=int(request.form['spd'])
    roc=int(request.form['roc'])
    k=int(request.form['k'])

    data_scaled=scaler.transform([[alt,spd,roc]])

    encoded_data=encoder.predict(data_scaled)
    kmeans=kmeans_models[k]
    phase=kmeans.predict(encoded_data)[0]

    def generate_plot():
        fig = plt.figure(figsize=(10,7))
        ax=fig.add_subplot(111,projection='3d')

        scatter=ax.scatter(data['alt'],data['spd'],data['roc'],
                             c=data[f'phase_K{k}'],cmap='viridis',alpha=0.6)

        ax.set_xlabel('Altitude')
        ax.set_ylabel('Speed')
        ax.set_zlabel('Rate of Climb')
        ax.set_title(f'K-Means Clustering of Flight Phases (K={k})')

        cbar=plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Phase')

        phases=data[f'phase_K{k}'].unique()
        colors=[scatter.cmap(scatter.norm(phase)) for phase in phases]
        patches=[mpatches.Patch(color=colors[i], label=f'Phase {phases[i]}') for i in range(len(phases))]
        plt.legend(handles=patches, title="Phases", loc='best')

        plt.savefig('static/plot.png')
        plt.close(fig)  

    generate_plot() 

    plot_url='static/plot.png'

    return render_template('result.html',phase=phase,plot_url=plot_url)


if __name__ == '__main__':
    app.run(debug=True)
