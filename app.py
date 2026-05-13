import streamlit as st
from deepface import DeepFace
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="MoodSyncAI",
    layout="wide"
)

st.title("🧠 MoodSyncAI")
st.subheader("Multi-Modal Sentiment & Emotion Analyzer")

st.write("""
This application performs:
- Facial emotion recognition
- Text sentiment analysis
- Emotion mismatch detection
- AI-generated emotional summaries
""")

# ---------------------------------
# LOAD SENTIMENT MODEL
# ---------------------------------

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_pipeline = load_sentiment_model()

# ---------------------------------
# USER INPUTS
# ---------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

user_text = st.text_area(
    "💬 Enter the sentence",
    "Everything is fine. I'm really happy with the results."
)

# ---------------------------------
# ANALYSIS BUTTON
# ---------------------------------

if st.button("🔍 Analyze Emotion"):

    if uploaded_file is None:
        st.error("Please upload a face image.")

    else:

        # ---------------------------------
        # DISPLAY IMAGE
        # ---------------------------------

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

        # ---------------------------------
        # FACIAL EMOTION ANALYSIS
        # ---------------------------------

        with st.spinner("Analyzing facial emotion..."):

            analysis = DeepFace.analyze(
                np.array(image),
                actions=['emotion'],
                enforce_detection=False
            )

            emotions = analysis[0]['emotion']
            dominant_emotion = analysis[0]['dominant_emotion']

        st.success("Facial emotion analysis completed!")

        # ---------------------------------
        # VISUAL EMOTION RESULTS
        # ---------------------------------

        st.subheader("📊 Visual Emotion Confidence")

        emotion_names = list(emotions.keys())
        emotion_values = list(emotions.values())

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(emotion_names, emotion_values)

        ax.set_xlabel("Emotion")
        ax.set_ylabel("Confidence")
        ax.set_title("Facial Emotion Detection")

        st.pyplot(fig)

        st.write(
            f"### Dominant Facial Emotion: **{dominant_emotion.upper()}**"
        )

        # ---------------------------------
        # TEXT SENTIMENT ANALYSIS
        # ---------------------------------

        with st.spinner("Analyzing text sentiment..."):

            sentiment_result = sentiment_pipeline(user_text)[0]

            text_sentiment = sentiment_result['label']
            text_confidence = round(
                sentiment_result['score'] * 100,
                2
            )

        st.subheader("📝 Text Sentiment Analysis")

        st.write(f"Sentiment: **{text_sentiment}**")
        st.write(f"Confidence: **{text_confidence}%**")

        # ---------------------------------
        # MULTIMODAL FUSION
        # ---------------------------------

        negative_emotions = [
            'sad',
            'fear',
            'angry',
            'disgust'
        ]

        mismatch = False

        if dominant_emotion in negative_emotions and text_sentiment == "POSITIVE":
            mismatch = True

        if dominant_emotion == "happy" and text_sentiment == "NEGATIVE":
            mismatch = True

        st.subheader("🔀 Fusion Result")

        if mismatch:
            st.warning("⚠️ MISMATCH DETECTED")
        else:
            st.success("✅ Emotional signals are aligned")

        # ---------------------------------
        # GENERATIVE EMOTIONAL SUMMARY
        # ---------------------------------

        st.subheader("🤖 Generative Emotional Summary")

        if mismatch:

            summary = (
                "Despite expressing positive sentiment verbally, "
                "the speaker's facial expression indicates possible "
                "stress, discomfort, or emotional tension. "
                "This emotional mismatch may suggest hidden feelings "
                "or emotional suppression during the conversation."
            )

        else:

            if text_sentiment == "POSITIVE":

                summary = (
                    "The facial expression and verbal sentiment are emotionally aligned, "
                    "indicating a positive and consistent emotional state."
                )

            else:

                summary = (
                    "The facial expression and verbal sentiment both indicate negative emotions, "
                    "suggesting emotional distress or dissatisfaction."
                )

        st.info(summary)

        # ---------------------------------
        # TECHNICAL DETAILS
        # ---------------------------------

        with st.expander("📚 Technical Details"):

            st.write("""
            ### Models Used

            - CNN Model: DeepFace Emotion Recognition
            - Transformer Model: DistilBERT
            - Fusion Layer: Rule-based multimodal fusion
            - Generative Component: AI-based emotional interpretation logic
            """)