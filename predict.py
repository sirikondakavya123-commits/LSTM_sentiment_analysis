import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200

# Load Model
model = load_model(
    "models/lstm_model.keras"
)

# Load Tokenizer
with open(
    "models/tokenizer.pkl",
    "rb"
) as f:
    tokenizer = pickle.load(f)

review = input(
    "Enter Review: "
)

sequence = tokenizer.texts_to_sequences(
    [review]
)

padded = pad_sequences(
    sequence,
    maxlen=MAX_LEN
)

prediction = model.predict(
    padded,
    verbose=0
)[0][0]

print(f"\nScore: {prediction:.4f}")

if prediction >= 0.5:
    print("Positive Review")
else:
    print("Negative Review")