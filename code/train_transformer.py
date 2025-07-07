
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load and prepare the data
def load_and_prepare_data(file_path='data/klax_daily.csv', sequence_length=14):
    df = pd.read_csv(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')

    # Normalize the data
    scaler = StandardScaler()
    df[['TMAX', 'TMIN']] = scaler.fit_transform(df[['TMAX', 'TMIN']])

    # Create sequences
    sequences = []
    targets = []
    for i in range(len(df) - sequence_length):
        sequences.append(df[['TMAX', 'TMIN']].iloc[i:i+sequence_length].values)
        targets.append(df['TMAX'].iloc[i+sequence_length])

    X = np.array(sequences)
    y = np.array(targets)

    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

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
        # For sequence-to-one prediction, we can use the output of the first token
        output = self.transformer(src, src)
        return self.fc_out(output[:, 0, :])

# Train the model
def train_model(X_train, y_train, model, epochs=50, batch_size=32, learning_rate=0.001):
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
def evaluate_model(X_test, y_test, model, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_test).float()).squeeze().numpy()
    
    # Inverse transform the predictions and actual values to get the real temperature values
    y_test_inv = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))]))[:, 0]
    predictions_inv = scaler.inverse_transform(np.hstack([predictions.reshape(-1, 1), np.zeros_like(predictions.reshape(-1, 1))]))[:, 0]

    mae = mean_absolute_error(y_test_inv, predictions_inv)
    print(f'Mean Absolute Error: {mae}')

# Inference function
def infer(model, data, scaler, sequence_length=14):
    model.eval()
    
    # Ensure data is a numpy array
    if isinstance(data, pd.DataFrame):
        data = data[['TMAX', 'TMIN']].values

    # Normalize the input data
    data_scaled = scaler.transform(data)
    
    # Reshape for the model
    data_tensor = torch.from_numpy(data_scaled).float().unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        prediction_scaled = model(data_tensor).squeeze().numpy()

    # Inverse transform the prediction
    prediction_inv = scaler.inverse_transform(np.hstack([prediction_scaled.reshape(-1, 1), np.zeros_like(prediction_scaled.reshape(-1, 1))]))[:, 0]
    
    return prediction_inv[0]

if __name__ == '__main__':
    (X_train, X_test, y_train, y_test), scaler = load_and_prepare_data()

    input_dim = 2
    model_dim = 32
    nhead = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 64

    model = TransformerModel(input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

    train_model(X_train, y_train, model)
    evaluate_model(X_test, y_test, model, scaler)

    # Save the model
    torch.save(model.state_dict(), 'model/klax_transformer_model.pth')

    # Example of how to use the infer function
    # Create some dummy data for inference
    dummy_data = pd.DataFrame({
        'TMAX': np.random.randint(60, 80, 14),
        'TMIN': np.random.randint(40, 60, 14)
    })
    prediction = infer(model, dummy_data, scaler)
    print(f'Example Inference Prediction: {prediction}')
