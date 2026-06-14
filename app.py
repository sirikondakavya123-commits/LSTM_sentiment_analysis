import streamlit as st
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200

# Load model
model = load_model(
    "models/lstm_model.keras"
)

# Load tokenizer
with open(
    "models/tokenizer.pkl",
    "rb"
) as f:
    tokenizer = pickle.load(f)

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬"
)

st.title(
    "🎬 IMDb Sentiment Analysis using LSTM"
)

review = st.text_area(
    "Enter Movie Review"
)

if st.button(
    "Predict Sentiment"
):

    if review.strip() == "":
        st.warning(
            "Please enter a review."
        )

    else:

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

        if prediction >= 0.5:
            st.success(
                f"😊 Positive Review ({prediction:.4f})"
            )
        else:
            st.error(
                f"😞 Negative Review ({prediction:.4f})"
            )

st.markdown("---")
st.write(
    "Built using TensorFlow, LSTM and Streamlit"
)