from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv("file_data.csv")

# Preprocess filepaths
def preprocess_filepath(filepath):
    filepath = re.sub(r'v\d+\.\d+', '<VERSION>', filepath)
    filepath = re.sub(r'product[A-Z]', '<PRODUCT>', filepath)
    return filepath

data['processed_filepath'] = data['filepath'].apply(preprocess_filepath)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  # Maximum sequence length

# Tokenize filepaths
def encode_filepaths(filepaths, tokenizer, max_len):
    return tokenizer(filepaths.tolist(), max_length=max_len, padding=True, truncation=True, return_tensors='tf')

encoded_data = encode_filepaths(data['processed_filepath'], tokenizer, max_len)
X = encoded_data['input_ids']
y = data['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=3, batch_size=16, validation_split=0.2)

# Evaluate model
y_pred = (model.predict(X_test).logits > 0).astype(int)
print(classification_report(y_test, y_pred))

# Save model
model.save_pretrained('bert_filepath_classifier')
tokenizer.save_pretrained('bert_filepath_tokenizer')




from transformers import BertTokenizer, TFBertForSequenceClassification

from BERT import preprocess_filepath

# Load model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('bert_filepath_classifier')
tokenizer = BertTokenizer.from_pretrained('bert_filepath_tokenizer')

# New filepath
new_filepath = "/src/productC_v3.0/module1/file4.txt"
processed = preprocess_filepath(new_filepath)
encoded = tokenizer(processed, max_length=max_len, padding=True, truncation=True, return_tensors='tf')
prediction = (model.predict(encoded['input_ids']).logits > 0).astype(int)
print("Flagged" if prediction[0] == 1 else "Not Flagged")