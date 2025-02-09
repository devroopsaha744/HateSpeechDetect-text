import streamlit as st
import requests
import json
from time import sleep
import plotly.graph_objects as go

# Configure the page
st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🛡️ Hate Speech Detection")
st.markdown("""
    This application analyzes text to detect potential hate speech and offensive language.
    Enter your text below to get started.
""")

# Text input
text_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Type or paste your text here..."
)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)

# Function to create gauge chart
def create_gauge_chart(prediction):
    severity_map = {
        "Hate Speech": 1.0,
        "Offensive Language": 0.6,
        "NOT Hate Speech": 0.2
    }
    
    color_map = {
        "Hate Speech": "red",
        "Offensive Language": "orange",
        "NOT Hate Speech": "green"
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = severity_map[prediction] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color_map[prediction]},
            'steps': [
                {'range': [0, 33], 'color': 'lightgreen'},
                {'range': [33, 66], 'color': 'lightyellow'},
                {'range': [66, 100], 'color': 'lightpink'}
            ],
        },
        title = {'text': "Content Safety Score"}
    ))
    
    fig.update_layout(height=250)
    return fig

if analyze_button and text_input:
    with st.spinner('Analyzing text...'):
        try:
            # Make API request
            response = requests.post(
                "https://datafreak-hatespeech-detect.hf.space/predict",
                json={"text": text_input}
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                
                # Display result with appropriate styling
                result_color = {
                    "Hate Speech": "red",
                    "Offensive Language": "orange",
                    "NOT Hate Speech": "green"
                }
                
                st.markdown(f"""
                    <div class="result-box" style="background-color: {result_color[prediction]}22;">
                        <h3 style="color: {result_color[prediction]};">Analysis Result: {prediction}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display gauge chart
                st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
                
                # Add explanation
                st.markdown("### Understanding the Result")
                if prediction == "NOT Hate Speech":
                    st.success("The content appears to be safe and appropriate.")
                elif prediction == "Offensive Language":
                    st.warning("The content contains offensive language. Consider revising for more appropriate communication.")
                else:
                    st.error("The content has been identified as hate speech. This type of content can be harmful and is often prohibited on many platforms.")
                
            else:
                st.error("Error: Unable to get prediction from the API.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
# Footer
st.markdown("---")
st.markdown("""
    ### About
    This tool uses a state-of-the-art machine learning model to detect potentially harmful content. 
    It can help moderators and users identify inappropriate content and maintain a safer online environment.
    
    **Note**: While this tool strives for accuracy, it should be used as an assistive tool rather than a definitive authority.
    Human judgment and context understanding are still essential in content moderation.
""")