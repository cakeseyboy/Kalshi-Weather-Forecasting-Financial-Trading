
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pickle
import asyncio
import os
import libsql_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Turso database configuration
DB_URL = os.getenv("TURSO_DATABASE_URL").replace("libsql", "https")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

# Load and prepare the data
async def load_and_prepare_data(sequence_length=14):
    # Load data sources
    daily_df = pd.read_csv('data/klax_daily.csv')

    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        context_rs = await client.execute("SELECT * FROM historical_nowcast_features")
        context_df = pd.DataFrame(context_rs.rows, columns=[col for col in context_rs.columns])
        
        marine_rs = await client.execute("SELECT * FROM historical_marine_features")
        marine_df = pd.DataFrame(marine_rs.rows, columns=[col for col in marine_rs.columns])

    # Convert DATE columns to datetime objects for merging and sorting
    daily_df['DATE'] = pd.to_datetime(daily_df['DATE'])
    context_df['DATE'] = pd.to_datetime(context_df['DATE'])
    marine_df['DATE'] = pd.to_datetime(marine_df['DATE'])

    # Explicitly cast 'sun_flag_8am' to float and fill NaNs with 0
    marine_df['sun_flag_8am'] = pd.to_numeric(marine_df['sun_flag_8am'], errors='coerce').fillna(0)

    # Drop 'dewpoint_spread_5am' from context_df to avoid column duplication/renaming during merge
    # We will use the 'dewpoint_spread_5am' from marine_df as it's part of the marine features.
    if 'dewpoint_spread_5am' in context_df.columns:
        context_df = context_df.drop(columns=['dewpoint_spread_5am'])

    # Merge dataframes
    # Start with daily_df as it contains TMAX (the target)
    df = pd.merge(daily_df, context_df, on='DATE', how='inner')
    df = pd.merge(df, marine_df, on='DATE', how='inner')

    df = df.sort_values('DATE')
    df.ffill(inplace=True) # Forward fill any missing values after merging
    df.bfill(inplace=True) # Backward fill any remaining missing values (e.g., at the beginning)

    # Define the expanded feature set
    features = [
        'TMAX', 'TMIN', 'avg_wind_speed_6h', 'avg_wind_direction_6h', 'morning_cloud_cover_avg', 
        'ceiling_slope', 'temp_slope_7_10am', 'sun_flag_8am', 'dewpoint_spread_5am'
    ]
    
    # Debugging prints
    print("\n--- Debugging df[features] before scaling ---")
    print("Data Types:\n", df[features].dtypes)
    print("Null Counts:\n", df[features].isnull().sum())
    print("--- End Debugging ---\n")

    # Normalize the data
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Create sequences
    sequences = []
    targets = []
    for i in range(len(df) - sequence_length):
        sequences.append(df[features].iloc[i:i+sequence_length].values)
        targets.append(df['TMAX'].iloc[i+sequence_length]) # Predict TMAX of the next day

    X = np.array(sequences)
    y = np.array(targets)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, X_val, y_train, y_val), scaler

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.src_embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, src):
        src = self.src_embedding(src) * np.sqrt(self.model_dim)
        output = self.transformer(src, src)
        return self.fc_out(output[:, 0, :])

# Train the model
def train_model(X_train, y_train, X_val, y_val, model, scaler, epochs=50, batch_size=32, learning_rate=0.001):
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                val_output = model(batch_X_val)
                val_loss = criterion(val_output, batch_y_val.unsqueeze(1))
                val_losses.append(val_loss.item())
                all_preds.extend(val_output.squeeze().numpy())
                all_targets.extend(batch_y_val.numpy())

        # De-normalize for MAE
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Create dummy arrays for inverse transform. The number of columns must match the scaler's original fit.
        # Here, it's the total number of features used for training.
        num_features = X_train.shape[2] # Get the actual number of features from the training data
        dummy_preds = np.zeros((len(all_preds), num_features))
        dummy_targets = np.zeros((len(all_targets), num_features))
        dummy_preds[:, 0] = all_preds
        dummy_targets[:, 0] = all_targets

        preds_denorm = scaler.inverse_transform(dummy_preds)[:, 0]
        targets_denorm = scaler.inverse_transform(dummy_targets)[:, 0]
        
        mae = mean_absolute_error(targets_denorm, preds_denorm)
        avg_val_loss = np.mean(val_losses)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, MAE: {mae:.4f}')


# Evaluate the model
def evaluate_model(X_test, y_test, model, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_test).float()).squeeze().numpy()
    
    # De-normalize for MAE
    num_features = X_test.shape[2] # Get the actual number of features from the test data
    dummy_preds = np.zeros((len(predictions), num_features))
    dummy_targets = np.zeros((len(y_test), num_features))
    dummy_preds[:, 0] = predictions
    dummy_targets[:, 0] = y_test

    predictions_inv = scaler.inverse_transform(dummy_preds)[:, 0]
    y_test_inv = scaler.inverse_transform(dummy_targets)[:, 0]

    mae = mean_absolute_error(y_test_inv, predictions_inv)
    print(f'Final Test MAE: {mae}')
    
    # Print one example prediction
    print(f"Example Prediction (de-normalized): {predictions_inv[0]:.2f}")
    print(f"Actual Value (de-normalized): {y_test_inv[0]:.2f}")


if __name__ == '__main__':
    (X_train, X_test, y_train, y_test), scaler = asyncio.run(load_and_prepare_data())

    # Update input_dim based on the new number of features
    input_dim = X_train.shape[2] # Dynamically get input_dim from the prepared data
    model_dim = 64
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 128
    dropout = 0.1

    model = TransformerModel(input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

    train_model(X_train, y_train, X_test, y_test, model, scaler, epochs=50, batch_size=32, learning_rate=0.001)
    evaluate_model(X_test, y_test, model, scaler)

    # Save the retrained model and scaler with _v2 suffix
    torch.save(model.state_dict(), 'model/klax_contextual_transformer_v2.pth')
    with open('model/contextual_scaler_v2.pkl', 'wb') as f:
        pickle.dump(scaler, f)
