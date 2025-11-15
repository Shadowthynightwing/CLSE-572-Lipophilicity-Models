import pandas as pd 
import os
import sys
from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import time

## DataSet Reading
try:
    directory_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    directory_path = os.getcwd()

Data = pd.read_csv(os.path.join(directory_path, 'Lipophilicity.csv'))
#display(Data)


## Scaling target with MinMaxScaler
scaler = MinMaxScaler()
exp_data = Data['exp'].values.reshape(-1,1) # Reshape for scaler

Data['exp_scaled'] = scaler.fit_transform(exp_data)
#display(Data.head())

# Importing the fingerprints.py file to generate fingerprints
from fingerprints import generate_morgan_fingerprint, generate_maccs_fingerprint

# Apply the functions to the dataframe
Data['Morgan_Fingerprint'] = Data['smiles'].apply(generate_morgan_fingerprint)
Data['MACCS_Keys'] = Data['smiles'].apply(generate_maccs_fingerprint)

# Display the updated dataframe with fingerprints
#display(Data.head())

## Creating Train-Test splits

#If fingerprint columns are missing or contain nulls, generate them from SMILES
if 'Morgan_Fingerprint' not in Data.columns or Data['Morgan_Fingerprint'].isnull().any():
    Data['Morgan_Fingerprint'] = Data['smiles'].apply(generate_morgan_fingerprint)
if 'MACCS_Keys' not in Data.columns or Data['MACCS_Keys'].isnull().any():
    Data['MACCS_Keys'] = Data['smiles'].apply(generate_maccs_fingerprint)

# Convert bitstring columns to numeric arrays (each string -> list of ints)
X_morgan = np.array([list(map(int, list(s))) for s in Data['Morgan_Fingerprint']])
X_maccs = np.array([list(map(int, list(s))) for s in Data['MACCS_Keys']])

Target = Data['exp_scaled'].values

# Split all arrays together so train/test rows remain aligned
X_train_morgan, X_test_morgan, X_train_maccs, X_test_maccs, Y_train, Y_test = train_test_split(
    X_morgan, X_maccs, Target, test_size=0.2, random_state=67)
#print("Shape of X_train_morgan:", X_train_morgan.shape)
#print("Shape of X_test_morgan:", X_test_morgan.shape)
#print("Shape of X_train_maccs:", X_train_maccs.shape)
#print("Shape of X_test_maccs:", X_test_maccs.shape)

#print("Shape of y_train:", Y_train.shape)
#print("Shape of y_test:", Y_test.shape)

## Regression Models
mlp_model = MLPRegressor(hidden_layer_sizes=(5000,), learning_rate=('adaptive'), max_iter=(1000), random_state=(47))


# Train Morgan Model
print("Morgan Model Training")
start_time1 = time.time()

mlp_model.fit(X_train_morgan, Y_train)

end_time1 = time.time()
training_time1 = end_time1 - start_time1
print(f"Training finished in {training_time1:.2f} seconds.")

Y_pred_morgan = mlp_model.predict(X_test_morgan)

# Train MACCS Model
print("MACCS Model Training")
start_time2 = time.time()

mlp_model.fit(X_train_maccs, Y_train)

end_time2 = time.time()
training_time2 = end_time2 - start_time2
print(f"Training finished in {training_time2:.2f} seconds.")

Y_pred_maccs = mlp_model.predict(X_test_maccs)


#print("Shape of y_pred_morgan:", Y_pred_morgan.shape)
#print("Data type of y_pred:", Y_pred_morgan.dtype)
#display(Y_pred_morgan[:10])

#print("Shape of y_pred_morgan:", Y_pred_maccs.shape)
#print("Data type of y_pred:", Y_pred_maccs.dtype)
#display(Y_pred_maccs[:10])

## Unscale the predictions
Y_pred_morgan_unscaled = scaler.inverse_transform(Y_pred_morgan.reshape(-1, 1))
Y_pred_maccs_unscaled = scaler.inverse_transform(Y_pred_maccs.reshape(-1, 1))
Y_test_unscaled = scaler.inverse_transform(Y_test.reshape(-1, 1))

## Metrics

# Compute RMSE using mean_squared_error with squared=False
Morgan_RMSE = root_mean_squared_error(Y_test_unscaled, Y_pred_morgan_unscaled)
MACCS_RMSE = root_mean_squared_error(Y_test_unscaled, Y_pred_maccs_unscaled)

print(f"Morgan RMSE: {Morgan_RMSE:.4f}")
print(f"MACCS RMSE: {MACCS_RMSE:.4f}")

# Displaying Conda Envi
os.getenv("CONDA_DEFAULT_ENV")
print(f"Conda Env Name: {os.getenv('CONDA_DEFAULT_ENV')}")

#Saving RMSE and Conda Env name to a text file
with open("Deliverables.csv", "w") as f:
    f.write(f"Conda Env Name: {os.getenv('CONDA_DEFAULT_ENV')}\n")
    f.write(f"Morgan RMSE: {Morgan_RMSE:.4f}\n")
    f.write(f"MACCS RMSE: {MACCS_RMSE:.4f}")