# app.py
# AI Medical Assistant with:
# - OpenRouter: Image analysis (vision)
# - NVIDIA NIM: Medical reasoning (Palmyra-Med)
# - ElevenLabs: AI speaks back like a real doctor
# - Voice input: You speak → AI hears
# - Text input: You type → AI reads
# - Model selector: Choose OpenRouter, NVIDIA NIM, or Both
# - AI speaks back like a doctor
# - Secure: Uses st.secrets for API keys

import os
import base64
import streamlit as st
from dotenv import load_dotenv
import requests
from openai import OpenAI  # For NVIDIA NIM API
from PIL import Image
from io import BytesIO
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pydub import AudioSegment
import speech_recognition as sr

# --- Load API Keys Securely (st.secrets for Streamlit Cloud, .env for local) ---
try:
    # ✅ On Streamlit Cloud: Use secrets.toml
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
    ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
except Exception as e:
    # 🖥️ Locally: Fall back to .env (for testing only)
    st.warning("⚠️ Running locally. Loading from .env file.")
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Validate keys
if not OPENROUTER_API_KEY:
    st.error("❌ OPENROUTER_API_KEY not found.")
    st.stop()
if not NVIDIA_API_KEY:
    st.error("❌ NVIDIA_API_KEY not found.")
    st.stop()
if not ELEVENLABS_API_KEY:
    st.error("❌ ELEVENLABS_API_KEY not found.")
    st.stop()

# --- 1. OpenRouter API (Vision) ---
openrouter_headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# --- 2. NVIDIA NIM API (Medical Expert: Palmyra-Med) ---
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# --- 3. ElevenLabs API (Realistic Doctor Voice) ---
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# App title
st.set_page_config(page_title="AI Doctor", layout="centered")
st.title("🩺 AI Doctor: Vision + Voice Assistant")
st.write("Upload a medical image and ask a question. The AI will respond like a real doctor.")

# --- 4. Upload Medical Image ---
uploaded_image = st.file_uploader(
    "📤 Upload a medical image (X-ray, MRI, skin rash, wound, etc.)",
    type=["png", "jpg", "jpeg"]
)

# --- 5. Show Uploaded Image ---
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Medical Image", use_container_width=True)

# --- 6. Speak Button for Voice Input ---
if st.button("🎤 Speak Your Question"):
    with st.spinner("🎙️ Listening... Please speak now."):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            user_question = recognizer.recognize_google(audio)
            st.success(f"✅ You said: *{user_question}*")

            # Save to session state
            st.session_state.user_question = user_question
            st.session_state.show_analyze = True

        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.warning("❌ Could not understand audio. Try speaking clearly.")
        except sr.RequestError:
            st.error("❌ Speech recognition service error.")

# --- 7. Text Input (Chatbot Style) ---
user_text = st.text_input(
    "💬 Or type your question",
    placeholder="E.g., What could this rash be? Is there pneumonia in this X-ray?"
)

if user_text:
    st.success(f"✅ You typed: *{user_text}*")
    st.session_state.user_question = user_text
    st.session_state.show_analyze = True

# --- 8. Show Model Selector & Analyze Button Only After Input ---
if 'show_analyze' in st.session_state and st.session_state.show_analyze and uploaded_image:
    model_choice = st.radio(
        "🧠 Choose AI Model for Response",
        options=[
            "OpenRouter (qwen/qwen-vl-plus) - Image Analysis",
            "NVIDIA NIM (Palmyra-Med) - Medical Expert",
            "Both (Compare Responses)"
        ],
        index=0,
        key="model_choice"
    )

    if st.button("🔍 Analyze Image & Respond"):
        with st.spinner("🧠 AI is analyzing the image..."):

            # --- Step 1: Use OpenRouter to analyze the image ---
            image_description = None
            if "OpenRouter" in model_choice or "Both" in model_choice:
                try:
                    uploaded_image.seek(0)
                    img_bytes = uploaded_image.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                    payload = {
                        "model": "qwen/qwen-vl-plus",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"Describe this image. User question: {st.session_state.user_question}"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"image/jpeg;base64,{img_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 500,
                        "temperature": 0.5
                    }

                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=openrouter_headers,
                        json=payload,
                        timeout=60
                    )
                    result = response.json()

                    if "error" in result:
                        st.error(f"❌ OpenRouter Error: {result['error']['message']}")
                        image_description = "Image could not be analyzed."
                    else:
                        image_description = result['choices'][0]['message']['content']
                        st.info("🖼️ Image Analysis (OpenRouter):")
                        st.write(image_description)

                except Exception as e:
                    st.error(f"❌ OpenRouter call failed: {str(e)}")
                    image_description = "Image could not be analyzed."

            # --- Step 2: Use Palmyra-Med for medical reasoning ---
            final_response = ""
            if "NVIDIA NIM" in model_choice or "Both" in model_choice:
                try:
                    med_response = nvidia_client.chat.completions.create(
                        model="writer/palmyra-med-70b-32k",
                        messages=[
                            {"role": "user", "content": f"""
                            You are a medical expert. Based on this image analysis, answer the user's question.

                            Image Analysis: {image_description}

                            User Question: {st.session_state.user_question}

                            Provide:
                            1. Likely conditions
                            2. Symptoms & causes
                            3. Treatment options
                            4. Clinical notes
                            5. Add: 'I am not a doctor. Please consult a licensed physician.'

                            Respond in a calm, professional tone.
                            """}
                        ],
                        max_tokens=800,
                        temperature=0.3,
                        top_p=0.7
                    )
                    final_response = med_response.choices[0].message.content
                    st.success("✅ Medical Expert Summary (Palmyra-Med):")
                    st.write(final_response)

                except Exception as e:
                    st.error(f"❌ Palmyra-Med Error: {str(e)}")
                    final_response = image_description or "No response available."

            # --- Step 3: Convert Final Response to Voice (Doctor-Like) ---
            if final_response.strip():
                with st.spinner("🔊 Converting response to speech..."):
                    try:
                        # Generate audio stream
                        audio_stream = elevenlabs_client.text_to_speech.convert(
                            voice_id="21m00Tcm4TlvDq8ikWAM",  # 👉 Rachel (calm, professional)
                            optimize_streaming_latency=1,
                            output_format="mp3_22050_32",
                            text=final_response,
                            model_id="eleven_multilingual_v2"
                        )

                        # Play audio
                        play(audio_stream)

                        # Show in Streamlit
                        with BytesIO() as audio_file:
                            for chunk in audio_stream:
                                audio_file.write(chunk)
                            audio_file.seek(0)
                            st.audio(audio_file.read(), format="audio/mp3")

                    except Exception as e:
                        st.error(f"❌ ElevenLabs error: {str(e)}")
                        st.warning("Falling back to gTTS...")
                        # Fallback to gTTS
                        try:
                            from gtts import gTTS
                            tts = gTTS(text=final_response, lang='en', slow=False)
                            audio_buffer = BytesIO()
                            tts.write_to_fp(audio_buffer)
                            audio_buffer.seek(0)
                            st.audio(audio_buffer, format="audio/mp3")
                            audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
                            play(audio_segment)
                        except Exception as e2:
                            st.error(f"❌ gTTS error: {str(e2)}")

# --- 9. Disclaimer ---
st.markdown("""
---
⚠️ **Disclaimer**: This app is for **educational purposes only**.  
It does **not** provide medical advice, diagnosis, or treatment.  
Always consult a licensed healthcare provider for medical concerns.
""", unsafe_allow_html=True)
