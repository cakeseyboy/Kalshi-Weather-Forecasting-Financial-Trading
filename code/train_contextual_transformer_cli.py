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

# Load and prepare the data with CLI targets
async def load_and_prepare_cli_data(sequence_length=14):
    """
    Load and prepare data for CLI settlement prediction.
    Joins historical_nowcast_features, historical_marine_features, and historical_cli_reports.
    """
    print("Loading data from Turso database...")
    
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        # Load all three tables
        rs_nowcast = await client.execute("SELECT * FROM historical_nowcast_features")
        nowcast_df = pd.DataFrame(rs_nowcast.rows, columns=[col for col in rs_nowcast.columns])
        
        rs_marine = await client.execute("SELECT * FROM historical_marine_features")
        marine_df = pd.DataFrame(rs_marine.rows, columns=[col for col in rs_marine.columns])
        
        rs_cli = await client.execute("SELECT * FROM historical_cli_reports")
        cli_df = pd.DataFrame(rs_cli.rows, columns=[col for col in rs_cli.columns])
    
    print(f"Loaded {len(nowcast_df)} nowcast records")
    print(f"Loaded {len(marine_df)} marine records") 
    print(f"Loaded {len(cli_df)} CLI records")
    
    # Convert DATE columns to datetime
    nowcast_df['DATE'] = pd.to_datetime(nowcast_df['DATE'])
    marine_df['DATE'] = pd.to_datetime(marine_df['DATE'])
    cli_df['DATE'] = pd.to_datetime(cli_df['DATE'])
    
    # Clean and prepare features
    marine_df['sun_flag_8am'] = pd.to_numeric(marine_df['sun_flag_8am'], errors='coerce').fillna(0)
    
    # Handle potential dewpoint_spread_5am duplication
    if 'dewpoint_spread_5am' in nowcast_df.columns:
        nowcast_df = nowcast_df.drop(columns=['dewpoint_spread_5am'])
    
    # Join all tables on DATE
    print("Joining tables...")
    df = pd.merge(cli_df, nowcast_df, on='DATE', how='inner')
    df = pd.merge(df, marine_df, on='DATE', how='inner')
    
    print(f"Final dataset: {len(df)} records after joins")
    
    # Sort by date and handle missing values
    df = df.sort_values('DATE')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Define features (excluding target variables)
    features = [
        'avg_wind_speed_6h', 'avg_wind_direction_6h', 'morning_cloud_cover_avg',
        'ceiling_slope', 'temp_slope_7_10am', 'sun_flag_8am', 'dewpoint_spread_5am'
    ]
    
    print("\n--- CLI Training Data Debug ---")
    print("Features:", features)
    print("Target: tmax_cli_f")
    print("Data Types:\n", df[features + ['tmax_cli_f']].dtypes)
    print("Null Counts:\n", df[features + ['tmax_cli_f']].isnull().sum())
    print("CLI Target Stats:")
    print(f"  Min: {df['tmax_cli_f'].min():.1f}°F")
    print(f"  Max: {df['tmax_cli_f'].max():.1f}°F") 
    print(f"  Mean: {df['tmax_cli_f'].mean():.1f}°F")
    print("--- End Debug ---\n")
    
    # Normalize only the features (not the target)
    scaler = StandardScaler()
    df_features_scaled = pd.DataFrame(
        scaler.fit_transform(df[features]), 
        columns=features,
        index=df.index
    )
    
    # Create sequences for transformer input
    sequences = []
    targets = []
    dates = []
    
    for i in range(len(df) - sequence_length):
        # Use 14 days of scaled features
        sequences.append(df_features_scaled.iloc[i:i+sequence_length].values)
        # Predict CLI TMAX for day 15
        targets.append(df['tmax_cli_f'].iloc[i+sequence_length])
        dates.append(df['DATE'].iloc[i+sequence_length])
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Created {len(X)} training sequences")
    print(f"Input shape: {X.shape}")  # Should be (samples, 14, 7)
    print(f"Target shape: {y.shape}")
    
    # Split data: 90% train, 10% test for robust evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    
    return (X_train, X_test, y_train, y_test), scaler

# Define the Transformer model (same architecture as v2)
class CliTransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(CliTransformerModel, self).__init__()
        self.model_dim = model_dim
        self.src_embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, src):
        src = self.src_embedding(src) * np.sqrt(self.model_dim)
        output = self.transformer(src, src)
        return self.fc_out(output[:, 0, :])  # Use first timestep output

# Train the CLI model with enhanced evaluation
def train_cli_model(X_train, y_train, X_test, y_test, model, epochs=150, batch_size=32, learning_rate=0.001):
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_mae = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation on test set
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            test_losses = []
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X_test, batch_y_test in test_loader:
                    test_output = model(batch_X_test)
                    test_loss = criterion(test_output.squeeze(), batch_y_test)
                    test_losses.append(test_loss.item())
                    all_preds.extend(test_output.squeeze().numpy())
                    all_targets.extend(batch_y_test.numpy())
            
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Calculate metrics
            mae = mean_absolute_error(all_targets, all_preds)
            
            # Calculate rounding accuracy (critical for Kalshi settlements)
            rounded_preds = np.round(all_preds)
            rounded_targets = np.round(all_targets)  # CLI targets should already be rounded
            rounding_accuracy = np.mean(rounded_preds == rounded_targets) * 100
            
            avg_train_loss = np.mean(train_losses)
            avg_test_loss = np.mean(test_losses)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')
            print(f'  MAE: {mae:.3f}°F | Rounding Accuracy: {rounding_accuracy:.1f}%')
            
            if mae < best_mae:
                best_mae = mae
                print(f'  *** New best MAE: {best_mae:.3f}°F ***')

# Evaluate the CLI model
def evaluate_cli_model(X_test, y_test, model):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_test).float()).squeeze().numpy()
    
    # Calculate comprehensive metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    
    # Rounding accuracy (most important for Kalshi)
    rounded_preds = np.round(predictions)
    rounded_targets = np.round(y_test)
    rounding_accuracy = np.mean(rounded_preds == rounded_targets) * 100
    
    # Within 1°F accuracy
    within_1f = np.mean(np.abs(y_test - predictions) <= 1.0) * 100
    
    print("\n=== CLI MODEL EVALUATION ===")
    print(f"Test MAE: {mae:.3f}°F")
    print(f"Test RMSE: {rmse:.3f}°F")
    print(f"Rounding Accuracy: {rounding_accuracy:.1f}%")
    print(f"Within 1°F: {within_1f:.1f}%")
    
    # Show example predictions
    print("\n=== Sample Predictions ===")
    for i in range(min(10, len(predictions))):
        print(f"Predicted: {predictions[i]:.1f}°F | Actual: {y_test[i]:.1f}°F | Rounded: {rounded_preds[i]:.0f}°F vs {rounded_targets[i]:.0f}°F")
    
    return mae, rounding_accuracy

if __name__ == '__main__':
    print("=== MARLIN v3: CLI Settlement Training ===")
    
    # Load CLI-targeted data
    (X_train, X_test, y_train, y_test), scaler = asyncio.run(load_and_prepare_cli_data())

    # Model hyperparameters (as specified)
    input_dim = 7  # 7 features
    model_dim = 64
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 128
    dropout = 0.1

    print(f"\n=== Model Architecture ===")
    print(f"Input dim: {input_dim}")
    print(f"Model dim: {model_dim}")
    print(f"Attention heads: {nhead}")
    print(f"Encoder layers: {num_encoder_layers}")
    print(f"Decoder layers: {num_decoder_layers}")

    # Initialize model
    model = CliTransformerModel(
        input_dim, model_dim, nhead, 
        num_encoder_layers, num_decoder_layers, 
        dim_feedforward, dropout
    )

    # Train the model
    print(f"\n=== Training CLI Model ===")
    train_cli_model(X_train, y_train, X_test, y_test, model, epochs=150, batch_size=32)
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    mae, accuracy = evaluate_cli_model(X_test, y_test, model)

    # Save the CLI model and scaler
    print(f"\n=== Saving CLI Model ===")
    torch.save(model.state_dict(), 'model/klax_contextual_transformer_cli.pth')
    with open('model/contextual_cli_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ CLI model saved as 'klax_contextual_transformer_cli.pth'")
    print("✅ CLI scaler saved as 'contextual_cli_scaler.pkl'")
    print(f"\n🎯 Final Performance: {mae:.3f}°F MAE | {accuracy:.1f}% Rounding Accuracy") 