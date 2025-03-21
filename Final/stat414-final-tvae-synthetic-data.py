import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TVAE(nn.Module):
    """
    Tabular Variational Autoencoder for synthetic data generation
    """
    def __init__(self, input_dim, latent_dim=16, hidden_dim=32):
        super(TVAE, self).__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


def preprocess_data(file_path, sample_fraction=0.1, random_state=42):
    """
    Load and preprocess the advertising data
    
    Args:
        file_path: Path to the CSV file
        sample_fraction: Fraction of data to sample (default: 0.1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_df, data_tensor, scaler)
    """
    logger.info(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise
    
    # Sample data
    df_sampled = df.sample(frac=sample_fraction, random_state=random_state)
    logger.info(f"Sampled {sample_fraction*100}% of data: {df_sampled.shape[0]} rows")
    
    # Save original column types for reconstruction later
    column_types = {col: df_sampled[col].dtype for col in df_sampled.columns}
    
    # Handle complex data types (lists, dates, etc.)
    for col in df_sampled.columns:
        if df_sampled[col].dtype == 'object':
            try:
                # Check if column contains date strings
                if df_sampled[col].str.contains('-').any() and df_sampled[col].str.contains(':').any():
                    logger.info(f"Converting date column {col} to numeric")
                    df_sampled[col] = pd.to_datetime(df_sampled[col]).astype(int) / 10**9
                else:
                    logger.info(f"Converting categorical column {col} to numeric")
                    df_sampled[col] = pd.factorize(df_sampled[col])[0]
            except:
                logger.info(f"Converting complex column {col} to numeric")
                df_sampled[col] = pd.factorize(df_sampled[col])[0]
    
    # Normalize numerical features
    logger.info("Normalizing numerical features...")
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_sampled)
    df_normalized = pd.DataFrame(normalized_data, columns=df_sampled.columns)
    
    # Convert normalized DataFrame to tensor
    data_tensor = torch.tensor(df_normalized.values, dtype=torch.float32)
    logger.info("Data converted to tensor for training")
    
    return df_sampled, data_tensor, scaler, column_types


def train_tvae(data_tensor, epochs=50, batch_size=64, learning_rate=0.001, latent_dim=16):
    """
    Train the TVAE model
    
    Args:
        data_tensor: Preprocessed data tensor
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        latent_dim: Dimension of the latent space
        
    Returns:
        Trained TVAE model
    """
    input_dim = data_tensor.shape[1]
    model = TVAE(input_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Create data loader
    dataset = torch.utils.data.TensorDataset(data_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Starting TVAE Training for {epochs} epochs...")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch[0])
            loss = loss_fn(output, batch[0])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    logger.info("Training completed!")
    return model


def generate_synthetic_data_by_label(model, df_sampled, scaler, column_types):
    """
    Generate synthetic data for each label class
    
    Args:
        model: Trained TVAE model
        df_sampled: Original sampled dataframe
        scaler: Fitted StandardScaler
        column_types: Original column data types
        
    Returns:
        DataFrame with synthetic data
    """
    latent_dim = model.encoder[-1].out_features
    synthetic_dfs = []
    
    # Ensure label column exists
    if "label" not in df_sampled.columns:
        raise ValueError("ERROR: Label column missing from dataset!")

    # Count real data for each label
    label_counts = df_sampled["label"].value_counts()
    logger.info(f"Label distribution in sampled data:\n{label_counts}")

    # Ensure at least two unique labels exist
    if len(label_counts) < 2:
        raise ValueError("ERROR: Not enough unique labels in sampled data!")

    # Get top 2 labels by frequency
    top_labels = label_counts.index[:2]
    label_A, label_B = top_labels
    logger.info(f"Generating synthetic data for labels: {label_A} and {label_B}")

    # Calculate synthetic data amounts
    label_A_real = int(label_counts[label_A])
    label_A_synthetic = int(label_A_real * 0.3)  # 30% of real data
    
    label_B_real = int(label_counts[label_B])
    label_B_synthetic = 100000  # Fixed amount for label B

    num_synthetic_per_label = {label_A: label_A_synthetic, label_B: label_B_synthetic}
    label_column_idx = df_sampled.columns.get_loc("label")

    for label, num_samples in num_synthetic_per_label.items():
        logger.info(f"Generating {num_samples} synthetic rows for label {label}...")

        if num_samples > 0:
            # Generate random points in latent space
            z = torch.randn((num_samples, latent_dim))
            
            # Decode to get synthetic data
            with torch.no_grad():
                synthetic_tensor = model.decoder(z)
            synthetic_data = synthetic_tensor.numpy()
            
            # Inverse transform to original scale
            synthetic_data = scaler.inverse_transform(synthetic_data)
            
            # Convert to DataFrame
            synthetic_df = pd.DataFrame(synthetic_data, columns=df_sampled.columns)
            
            # Ensure label values are correct (force correct label)
            if label == label_B:
                synthetic_df["label"] = label_B
            else:
                synthetic_df["label"] = label_A
            
            # Post-process to match original data types
            for col, dtype in column_types.items():
                if pd.api.types.is_integer_dtype(dtype):
                    synthetic_df[col] = synthetic_df[col].round().astype(int)
            
            synthetic_dfs.append(synthetic_df)
            logger.info(f"Generated {num_samples} rows for label {label}")

    # Combine all synthetic data
    return pd.concat(synthetic_dfs, ignore_index=True)


def main():
    # Input and output file paths
    input_file = "train_data_ads.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"synthetic_ads_data_{timestamp}.csv"
    
    # Parameters
    sample_fraction = 0.1  # Use 10% of original data
    epochs = 50
    batch_size = 64
    latent_dim = 16
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    try:
        # Preprocess data
        df_sampled, data_tensor, scaler, column_types = preprocess_data(
            input_file, sample_fraction=sample_fraction
        )
        
        # Train model
        model = train_tvae(
            data_tensor, 
            epochs=epochs, 
            batch_size=batch_size, 
            latent_dim=latent_dim
        )
        
        # Generate synthetic data
        logger.info("Generating synthetic data...")
        synthetic_df = generate_synthetic_data_by_label(model, df_sampled, scaler, column_types)
        
        # Merge real and synthetic data
        final_dataset = pd.concat([df_sampled, synthetic_df], ignore_index=True)
        logger.info(f"Synthetic data merged! Final dataset has {len(final_dataset)} rows")
        
        # Save model and final dataset
        torch.save(model.state_dict(), os.path.join("output", f"tvae_model_{timestamp}.pt"))
        final_dataset.to_csv(os.path.join("output", output_file), index=False)
        logger.info(f"Results saved to output/{output_file}")
        
    except Exception as e:
        logger.error(f"Error in synthetic data generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
