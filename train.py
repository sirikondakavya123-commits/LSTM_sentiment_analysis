import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Parameters
VOCAB_SIZE = 10000
MAX_LEN = 200

# Load Dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Convert labels
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE)

tokenizer.fit_on_texts(df["review"])

X = tokenizer.texts_to_sequences(df["review"])

X = pad_sequences(
    X,
    maxlen=MAX_LEN
)

y = df["sentiment"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build LSTM Model
model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128
    ),

    LSTM(64),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Save model
model.save("models/lstm_model.keras")

# Save tokenizer
with open(
    "models/tokenizer.pkl",
    "wb"
) as f:
    pickle.dump(tokenizer, f)

print("Model Saved Successfully")