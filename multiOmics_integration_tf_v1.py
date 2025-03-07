import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class Sampling(layers.Layer):
    """Reparameterization trick by sampling from an isotropic normal distribution."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class MultiOmicsVAE(Model):
    def __init__(self, rnaseq_dim, metag_dim, hidden_dim=128, latent_dim=32, name="multi_omics_vae", **kwargs):
        super(MultiOmicsVAE, self).__init__(name=name, **kwargs)
        
        self.rnaseq_dim = rnaseq_dim
        self.metag_dim = metag_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # RNA-seq encoder
        self.rnaseq_encoder = keras.Sequential([
            layers.InputLayer(input_shape=(rnaseq_dim,)),
            layers.Dense(hidden_dim),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(hidden_dim // 2),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        
        # Metagenomics encoder
        self.metag_encoder = keras.Sequential([
            layers.InputLayer(input_shape=(metag_dim,)),
            layers.Dense(hidden_dim),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(hidden_dim // 2),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        
        # Joint latent space layers
        self.concat_layer = layers.Concatenate()
        self.dense_joint = layers.Dense(hidden_dim)
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        
        # RNA-seq decoder
        self.rnaseq_decoder = keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(hidden_dim // 2),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(hidden_dim),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(rnaseq_dim),
        ])
        
        # Metagenomics decoder
        self.metag_decoder = keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(hidden_dim // 2),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(hidden_dim),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(metag_dim),
        ])
        
        # Define total loss tracker and training step counter
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.rnaseq_loss_tracker = keras.metrics.Mean(name="rnaseq_loss")
        self.metag_loss_tracker = keras.metrics.Mean(name="metag_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rnaseq_loss_tracker,
            self.metag_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def encode(self, rnaseq_data, metag_data, training=False):
        # Encode each data type
        rnaseq_h = self.rnaseq_encoder(rnaseq_data, training=training)
        metag_h = self.metag_encoder(metag_data, training=training)
        
        # Combine features for joint latent space
        combined_h = self.concat_layer([rnaseq_h, metag_h])
        joint_h = self.dense_joint(combined_h)
        
        # Get latent parameters
        z_mean = self.z_mean(joint_h)
        z_log_var = self.z_log_var(joint_h)
        
        return z_mean, z_log_var
    
    def decode(self, z, training=False):
        # Decode to get reconstructions of both data types
        rnaseq_recon = self.rnaseq_decoder(z, training=training)
        metag_recon = self.metag_decoder(z, training=training)
        
        return rnaseq_recon, metag_recon
    
    def call(self, inputs, training=False):
        rnaseq_data, metag_data = inputs
        
        # Encode
        z_mean, z_log_var = self.encode(rnaseq_data, metag_data, training=training)
        
        # Sample from latent space
        z = self.sampling([z_mean, z_log_var])
        
        # Decode
        rnaseq_recon, metag_recon = self.decode(z, training=training)
        
        return rnaseq_recon, metag_recon, z_mean, z_log_var
    
    def train_step(self, data):
        if isinstance(data, tuple):
            rnaseq_data, metag_data = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            rnaseq_recon, metag_recon, z_mean, z_log_var = self([rnaseq_data, metag_data], training=True)
            
            # Compute losses
            rnaseq_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(rnaseq_data, rnaseq_recon),
                    axis=1
                )
            )
            
            metag_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(metag_data, metag_recon),
                    axis=1
                )
            )
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            # Total loss (beta is the weight of KL term - can be adjusted)
            beta = self.beta
            total_loss = rnaseq_loss + metag_loss + beta * kl_loss
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.rnaseq_loss_tracker.update_state(rnaseq_loss)
        self.metag_loss_tracker.update_state(metag_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "rnaseq_loss": self.rnaseq_loss_tracker.result(),
            "metag_loss": self.metag_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        if isinstance(data, tuple):
            rnaseq_data, metag_data = data
        
        # Forward pass
        rnaseq_recon, metag_recon, z_mean, z_log_var = self([rnaseq_data, metag_data], training=False)
        
        # Compute losses
        rnaseq_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(rnaseq_data, rnaseq_recon),
                axis=1
            )
        )
        
        metag_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(metag_data, metag_recon),
                axis=1
            )
        )
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        
        # Total loss
        beta = self.beta
        total_loss = rnaseq_loss + metag_loss + beta * kl_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.rnaseq_loss_tracker.update_state(rnaseq_loss)
        self.metag_loss_tracker.update_state(metag_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "rnaseq_loss": self.rnaseq_loss_tracker.result(),
            "metag_loss": self.metag_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def load_and_preprocess_data(rnaseq_adata, metag_adata):
    """
    Assumes both AnnData objects have the same observations (samples) in the same order
    Returns preprocessed data ready for the VAE
    """
    # Extract the matrices
    rnaseq_data = rnaseq_adata.X.copy()
    metag_data = metag_adata.X.copy()
    
    # Convert to dense if sparse
    if isinstance(rnaseq_data, np.ndarray) == False:
        rnaseq_data = rnaseq_data.toarray()
    if isinstance(metag_data, np.ndarray) == False:
        metag_data = metag_data.toarray()
    
    # Split data into train/test sets
    rnaseq_train, rnaseq_test, metag_train, metag_test = train_test_split(
        rnaseq_data, metag_data, test_size=0.2, random_state=42
    )
    
    # Get dimensions
    rnaseq_dim = rnaseq_data.shape[1]
    metag_dim = metag_data.shape[1]
    
    return (rnaseq_train, metag_train), (rnaseq_test, metag_test), rnaseq_dim, metag_dim

def extract_latent_features(model, data):
    """
    Extract latent space features for all samples
    """
    rnaseq_data, metag_data = data
    z_mean, _ = model.encode(rnaseq_data, metag_data, training=False)
    return z_mean.numpy()

def feature_importance(model):
    """
    Estimate feature importance by analyzing decoder weights
    """
    # Get the last layer weights for RNA-seq decoder
    rnaseq_decoder_weights = model.rnaseq_decoder.layers[-1].get_weights()[0]
    
    # Get the last layer weights for metagenomics decoder
    metag_decoder_weights = model.metag_decoder.layers[-1].get_weights()[0]
    
    # Calculate importance score (sum of absolute weights for each feature)
    rnaseq_importance = np.sum(np.abs(rnaseq_decoder_weights), axis=0)
    metag_importance = np.sum(np.abs(metag_decoder_weights), axis=0)
    
    return rnaseq_importance, metag_importance

def plot_loss(history):
    """
    Plot training loss curves
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['rnaseq_loss'])
    plt.plot(history.history['val_rnaseq_loss'])
    plt.title('RNA-seq Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(2, 2, 3)
    plt.plot(history.history['metag_loss'])
    plt.plot(history.history['val_metag_loss'])
    plt.title('Metagenomic Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(2, 2, 4)
    plt.plot(history.history['kl_loss'])
    plt.plot(history.history['val_kl_loss'])
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_latent_space(latent_features, metadata=None, color_by=None):
    """
    Plot the first two dimensions of the latent space
    """
    plt.figure(figsize=(10, 8))
    
    if metadata is not None and color_by is not None and color_by in metadata.columns:
        sns.scatterplot(x=latent_features[:, 0], y=latent_features[:, 1], 
                        hue=metadata[color_by], palette='viridis')
        plt.legend(title=color_by)
    else:
        plt.scatter(latent_features[:, 0], latent_features[:, 1])
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space')
    plt.grid(True)
    plt.show()

def visualize_feature_importance(rnaseq_importance, metag_importance, rnaseq_feature_names, metag_feature_names, top_n=20):
    """
    Visualize the top most important features from each data type
    """
    # RNA-seq features
    top_rnaseq_idx = np.argsort(rnaseq_importance)[::-1][:top_n]
    top_rnaseq_importance = rnaseq_importance[top_rnaseq_idx]
    top_rnaseq_names = [rnaseq_feature_names[i] for i in top_rnaseq_idx]
    
    # Metagenomics features
    top_metag_idx = np.argsort(metag_importance)[::-1][:top_n]
    top_metag_importance = metag_importance[top_metag_idx]
    top_metag_names = [metag_feature_names[i] for i in top_metag_idx]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # RNA-seq
    sns.barplot(x=top_rnaseq_importance, y=top_rnaseq_names, ax=ax1)
    ax1.set_title(f'Top {top_n} RNA-seq Features')
    ax1.set_xlabel('Importance Score')
    
    # Metagenomics
    sns.barplot(x=top_metag_importance, y=top_metag_names, ax=ax2)
    ax2.set_title(f'Top {top_n} Metagenomic Features')
    ax2.set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.show()

# Custom callback for adjusting the beta value during training
class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, start_beta=0.0, target_beta=1.0, n_epochs=10, verbose=1):
        super(BetaScheduler, self).__init__()
        self.start_beta = start_beta
        self.target_beta = target_beta
        self.n_epochs = n_epochs
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.n_epochs:
            new_beta = self.start_beta + (self.target_beta - self.start_beta) * (epoch / self.n_epochs)
        else:
            new_beta = self.target_beta
            
        self.model.beta = new_beta
        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: beta = {new_beta:.4f}")

# Main execution
def run_multi_omics_integration(rnaseq_adata, metag_adata, hidden_dim=128, latent_dim=32, 
                             num_epochs=100, beta=1.0, lr=1e-3, batch_size=32):
    """
    Main function to run the multi-omics integration pipeline using TensorFlow
    
    Parameters:
    - rnaseq_adata: AnnData object with RNA-seq data
    - metag_adata: AnnData object with metagenomic data
    - hidden_dim: Dimension of hidden layers
    - latent_dim: Dimension of latent space
    - num_epochs: Number of training epochs
    - beta: Weight for KL divergence term
    - lr: Learning rate
    - batch_size: Batch size for training
    
    Returns:
    - model: Trained VAE model
    - latent_features: Extracted latent features for all samples
    - rnaseq_importance: Feature importance scores for RNA-seq features
    - metag_importance: Feature importance scores for metagenomic features
    """
    # Check if samples match
    assert rnaseq_adata.obs_names.equals(metag_adata.obs_names), "Sample IDs don't match between datasets"
    
    # Load and preprocess data
    (rnaseq_train, metag_train), (rnaseq_test, metag_test), rnaseq_dim, metag_dim = load_and_preprocess_data(
        rnaseq_adata, metag_adata
    )
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((rnaseq_train, metag_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(rnaseq_train)).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((rnaseq_test, metag_test))
    test_dataset = test_dataset.batch(batch_size)
    
    # Initialize model
    model = MultiOmicsVAE(rnaseq_dim, metag_dim, hidden_dim, latent_dim)
    model.beta = beta  # Set initial beta value
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer)
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        BetaScheduler(start_beta=0.0, target_beta=beta, n_epochs=10)
    ]
    
    # Train model
    print("Training VAE model...")
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset,
        callbacks=callbacks
    )
    
    # Plot loss curves
    plot_loss(history)
    
    # Extract latent features for all data
    print("Extracting latent features...")
    full_rnaseq_data = rnaseq_adata.X.toarray() if hasattr(rnaseq_adata.X, 'toarray') else rnaseq_adata.X
    full_metag_data = metag_adata.X.toarray() if hasattr(metag_adata.X, 'toarray') else metag_adata.X
    full_data = (full_rnaseq_data, full_metag_data)
    
    latent_features = extract_latent_features(model, full_data)
    
    # Get feature importance
    print("Calculating feature importance...")
    rnaseq_importance, metag_importance = feature_importance(model)
    
    # Plot latent space
    if hasattr(rnaseq_adata, 'obs') and isinstance(rnaseq_adata.obs, pd.DataFrame):
        if 'group' in rnaseq_adata.obs.columns:
            plot_latent_space(latent_features, rnaseq_adata.obs, 'group')
        else:
            plot_latent_space(latent_features, rnaseq_adata.obs)
    else:
        plot_latent_space(latent_features)
    
    # Plot feature importance
    if hasattr(rnaseq_adata, 'var_names') and hasattr(metag_adata, 'var_names'):
        visualize_feature_importance(rnaseq_importance, metag_importance, 
                                     rnaseq_adata.var_names, metag_adata.var_names)
    
    # Add latent features to AnnData objects
    latent_df = pd.DataFrame(latent_features, index=rnaseq_adata.obs_names,
                             columns=[f'latent_{i}' for i in range(latent_dim)])
    
    rnaseq_adata.obsm['X_vae'] = latent_features
    metag_adata.obsm['X_vae'] = latent_features
    
    return model, latent_features, rnaseq_importance, metag_importance, latent_df

# Example usage:
if __name__ == "__main__":
    # Simulated data creation for demonstration
    
    # Create simulated RNA-seq data
    n_samples = 100
    n_genes = 1000
    n_taxa = 500
    
    # Simulated RNA-seq data
    rna_data = np.random.negative_binomial(10, 0.5, size=(n_samples, n_genes))
    rna_adata = sc.AnnData(rna_data)
    rna_adata.var_names = [f'gene_{i}' for i in range(n_genes)]
    rna_adata.obs_names = [f'sample_{i}' for i in range(n_samples)]
    
    # Simulated metagenomic data
    meta_data = np.random.negative_binomial(5, 0.7, size=(n_samples, n_taxa))
    meta_adata = sc.AnnData(meta_data)
    meta_adata.var_names = [f'taxa_{i}' for i in range(n_taxa)]
    meta_adata.obs_names = [f'sample_{i}' for i in range(n_samples)]
    
    # Add some group labels for visualization
    groups = np.random.choice(['A', 'B', 'C'], size=n_samples)
    rna_adata.obs['group'] = groups
    meta_adata.obs['group'] = groups
    
    # Normalize data
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    
    sc.pp.normalize_total(meta_adata, target_sum=1e4)
    sc.pp.log1p(meta_adata)
    
    # Run multi-omics integration
    model, latent_features, rna_importance, meta_importance, latent_df = run_multi_omics_integration(
        rna_adata, meta_adata, hidden_dim=128, latent_dim=20, num_epochs=50, beta=0.5
    )
    
    print("Integration complete!")
    print(f"Extracted {latent_features.shape[1]} latent dimensions from {n_samples} samples")

