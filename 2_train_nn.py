
# 2_train_nn.py (AlphaZero-Style Residual Network)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from pathlib import Path
import sys

# --- CONFIGURATION ---
DATASET_NPZ_FILE = "training_data.npz"
MODEL_H5_FILE = "chess_nn_model_resnet.h5" # Renamed model for clarity
RESIDUAL_BLOCKS = 3 # Number of blocks to use
# ---------------------

def load_data():
    """Loads and reshapes the dataset for Keras."""
    if not Path(DATASET_NPZ_FILE).exists():
        print(f"Error: Run 1_convert_data.py first to create {DATASET_NPZ_FILE}.")
        sys.exit(1)
        
    print("Loading dataset...")
    data = np.load(DATASET_NPZ_FILE)
    X = data['X'].transpose(0, 2, 3, 1) 
    Y = data['Y']
    
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    print(f"Loaded {len(X)} total samples. Training: {len(X_train)}, Validation: {len(X_val)}")
    return X_train, Y_train, X_val, Y_val

# --- RESIDUAL BLOCK DEFINITION ---
def residual_block(x, filters, kernel_size=(3, 3), l2_reg=0.0001):
    """A standard ResNet block with convolution, batch normalization, and skip connection."""
    
    # Save input for skip connection
    shortcut = x 
    
    # 1. Convolution Layer
    x = layers.Conv2D(filters, kernel_size, padding='same', 
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 2. Convolution Layer (Returns to the original number of filters)
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    
    # Skip Connection: Add the original input (shortcut) to the output
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def build_resnet_model(num_blocks=RESIDUAL_BLOCKS, filters=128):
    """
    Builds the complete AlphaZero-style Residual Network.
    """
    l2_reg = 0.0001
    
    # 1. Input Layer (8x8x12)
    input_layer = keras.Input(shape=(8, 8, 12))
    
    # 2. Initial Convolution Block
    x = layers.Conv2D(filters, (3, 3), padding='same', 
                      kernel_regularizer=regularizers.l2(l2_reg))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 3. Residual Tower
    for _ in range(num_blocks):
        x = residual_block(x, filters=filters, l2_reg=l2_reg)
    
    # 4. Global Average Pooling (Compresses 8x8 to a single vector)
    x = layers.GlobalAveragePooling2D()(x)
    
    # 5. Value Head (Output)
    # The Value Head is simpler in ResNet, aiming for direct score prediction.
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # Output: Single score from -1.0 to 1.0 (Value)
    output_layer = layers.Dense(1, activation='tanh', name='value_head')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Use a low, stable learning rate
    custom_adam = keras.optimizers.Adam(learning_rate=0.0005) 
    
    model.compile(optimizer=custom_adam, loss='mse', metrics=['mae'])
    model.summary()
    return model

def train_model(model, X_train, Y_train, X_val, Y_val):
    print("Starting training...")
    callbacks = [
        # Restore the best weights based on validation loss
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) 
    ]
    
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=30, # Set high, Early Stopping manages runtime
        batch_size=64, 
        callbacks=callbacks
    )
    
    model.save(MODEL_H5_FILE)
    print(f"\nModel trained and saved to {MODEL_H5_FILE}")

if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = load_data()
    model = build_resnet_model()
    train_model(model, X_train, Y_train, X_val, Y_val)
EOF