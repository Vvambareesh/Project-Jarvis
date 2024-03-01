import os
import numpy as np
from tensorflow import keras 
from tensorflow.keras import layers

# Directories
SPECTROGRAM_DIR = "spectrograms" 
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

# Load spectrograms
train_spectrograms = []
for img_file in os.listdir(os.path.join(SPECTROGRAM_DIR, TRAIN_DIR)):
  img_path = os.path.join(SPECTROGRAM_DIR, TRAIN_DIR, img_file)
  img = keras.preprocessing.image.load_img(img_path, grayscale=True)
  train_spectrograms.append(img)

train_spectrograms = np.array(train_spectrograms) / 255.0

# Similarly load validation and test spectrograms

# Load transcripts 
train_transcripts = []
for txt_file in os.listdir(os.path.join(TRAIN_DIR)):
  with open(os.path.join(TRAIN_DIR, txt_file)) as f:
    train_transcripts.append(f.read())
    
# Character mappings
chars = set(''.join(train_transcripts))
char_to_idx = {c:i for i, c in enumerate(chars)} 
idx_to_char = {i:c for i, c in enumerate(chars)}

# Vectorize transcripts 
train_labels = [[char_to_idx[c] for c in t] for t in train_transcripts]

# Model architecture  
input_shape = train_spectrograms[0].shape
output_size = len(char_to_idx)

inp = layers.Input(shape=input_shape)
...
out = layers.Dense(output_size)(...) 

model = keras.Model(inp, out)

# Compile & train 
model.compile(...)  
model.fit(train_spectrograms, train_labels)