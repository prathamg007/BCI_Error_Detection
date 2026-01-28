from sklearn.neural_network import MLPClassifier
from src import config

def build_model():
    """
    Constructs an MLP Classifier optimized for EEG Signal Classification.
    Architecture: Input -> Dense(100) -> Dense(50) -> Output
    """
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.01,
        batch_size=config.BATCH_SIZE,
        learning_rate_init=config.LEARNING_RATE,
        max_iter=config.EPOCHS,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True
    )
    return model