import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
LABELS_FILE = os.path.join(DATA_DIR, 'TrainLabels.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'bci_mlp_model.pkl')

# Signal Processing Constants
FS = 200            # Sampling Frequency (Hz)
WINDOW_SEC = 1.3    # Epoch duration (seconds)
DOWNSAMPLE_RATE = 4 # Keep 1 every 4 samples to save RAM
LOW_CUT = 1.0       # Bandpass Filter Low Cutoff (Hz)
HIGH_CUT = 40.0     # Bandpass Filter High Cutoff (Hz)

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 300