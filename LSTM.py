import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv("file_data.csv")  # Columns: filepath, label

# Preprocess filepaths
def preprocess_filepath(filepath):
    filepath = re.sub(r'v\d+\.\d+', '<VERSION>', filepath)  # Normalize versions
    filepath = re.sub(r'product[A-Z]', '<PRODUCT>', filepath)  # Normalize products
    return filepath

data['processed_filepath'] = data['filepath'].apply(preprocess_filepath)

# Tokenize filepaths
max_words = 5000  # Maximum vocabulary size
max_len = 50  # Maximum sequence length
tokenizer = Tokenizer(num_words=max_words, filters='', lower=False, split='/')
tokenizer.fit_on_texts(data['processed_filepath'])
X = tokenizer.texts_to_sequences(data['processed_filepath'])
X = pad_sequences(X, maxlen=max_len, padding='post')
y = data['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Build LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weight_dict)

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Save model and tokenizer
model.save('filepath_classifier.h5')
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
    
    


import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('filepath_classifier.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# New filepath
new_filepath = "/src/productC_v3.0/module1/file4.txt"
processed = preprocess_filepath(new_filepath)
sequence = tokenizer.texts_to_sequences([processed])
padded = pad_sequences(sequence, maxlen=max_len, padding='post')
prediction = (model.predict(padded) > 0.5).astype(int)
print("Flagged" if prediction[0] == 1 else "Not Flagged")