import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astrapy import DataAPIClient
import plotly.express as px
from wordcloud import WordCloud
from models import run_flow , photo_flow , video_flow
import os 
from PIL import Image 
import base64

# Initialize the Astra client
client = DataAPIClient("AstraCS:zgJSwvWzsZXjawoPFyNcXMNS:9eed3b61fbbf2bf615a766b0a661bf597de4340be47436100e3e84fd1b38fa14")

db = client.get_database_by_api_endpoint(
    "https://fb32a9e0-4a7c-48a5-9b2b-0a993c2abdf4-westus3.apps.astra.datastax.com"
)
social_collection = db["some_data"]
documents = social_collection.find({})
data = list(documents)
df = pd.DataFrame(data)

# Ensure numeric columns have the correct types
INT_Cols = ['Likes', 'Comments', 'Shares', 'Impressions', 'Reach', 'Audience_Age']
Float_Cols = ['Engagement_Rate']

for col in INT_Cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in Float_Cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if "Post_Timestamp" in df.columns:
    # Attempt to infer format and handle missing/invalid timestamps
    df["Post_Timestamp"] = pd.to_datetime(
        df["Post_Timestamp"], errors="coerce"
    )
    
    # Drop rows where parsing failed (NaT entries)
    df = df.dropna(subset=["Post_Timestamp"])

    # Sort by timestamp if needed
    df = df.sort_values(by="Post_Timestamp")

# Define a function to simulate chatbot interaction
def chatbot_response(user_input, file=None):
    """Simulate a chatbot response based on user input."""
    # Example simulated response
    response = {
        "engagement": {
            "likes": np.random.randint(50, 500, 10),
            "comments": np.random.randint(10, 200, 10),
            "shares": np.random.randint(5, 100, 10),
            "dates": pd.date_range(start="2023-01-01", periods=10).tolist(),
        },
        "summary": "Your engagement has been consistent, with an average of 250 likes, 50 comments, and 30 shares per post."
    }
    return response

# Streamlit app setup
st.set_page_config(page_title="Chai Analytics", layout="wide", page_icon="📊")

# App title and description
_, col2, _ = st.columns([1, 2, 1])
with col2:
    st.title(" Chai Analytics")

# Page navigation
page = st.sidebar.selectbox("Choose a page:", ["Chat Interaction", "Engagement Analytics" , "image analytics" , "reel analytics"])

if page == "image analytics":
    st.markdown(
        """
        <style>
        .fixed-height {
            height: 800px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .uploaded-image {
            height: 800px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    def image_to_base64(image_path):
        """Convert image to Base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.header("Chatbot Interaction")

    # Upload file section
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Save the uploaded image as uploads/photo.jpg
        image_path = "uploads/photo.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load the image for display
        image = Image.open(image_path)

        # Analyze the image
        answer = photo_flow(image_path)

        # Convert the image to Base64
        base64_image = image_to_base64(image_path)

        # Layout the UI
        col1, col2 = st.columns(2, gap="large")

        # Display image in the first column
        with col1:
            st.markdown("### Uploaded Image")
            st.markdown(
                f"<div class='fixed-height'><img class='uploaded-image' src='data:image/jpeg;base64,{base64_image}' /></div>",
                unsafe_allow_html=True
            )

        # Display markdown answer in the second column
        with col2:
            st.markdown("### Analysis Result")
            st.markdown(
                f"<div class='fixed-height'>{answer.text}</div>",
                unsafe_allow_html=True
            )
if page == "reel analytics":
    st.markdown(
        """
        <style>
        .fixed-height {
            height: 800px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .uploaded-video {
            height: 800px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    def video_to_base64(video_path):
        """Convert video to Base64 string."""
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")

    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.header("Chatbot Interaction - reel analytics")

    # Upload file section
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Save the uploaded video as uploads/video.mp4
        video_path = "uploads/video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Analyze the video
        answer = video_flow(video_path)

        # Convert the video to Base64
        base64_video = video_to_base64(video_path)

        # Layout the UI
        col1, col2 = st.columns(2, gap="large")

        # Display video in the first column
        with col1:
            st.markdown("### Uploaded Video")
            st.markdown(
                f"<div class='fixed-height'><video class='uploaded-video' controls><source src='data:video/mp4;base64,{base64_video}' type='video/mp4'>Your browser does not support the video tag.</video></div>",
                unsafe_allow_html=True
            )

        # Display markdown answer in the second column
        with col2:
            st.markdown("### Analysis Result")
            st.markdown(
                f"<div class='fixed-height'>{answer.text}</div>",
                unsafe_allow_html=True
            )

if page == "Chat Interaction":
    st.write(' ')
    st.write(' ')
    st.write(' ')
    # Chat interaction page
    st.header("Chatbot Interaction")
    st.write(' ')
    st.write(' ')
    user_input = st.text_input("Type your message here:", placeholder="Ask about your account analytics...")

    if st.button("Send"):
        if user_input.strip():
            with st.spinner("Chatbot is analyzing your input..."):
                # Calling the custom function and storing the result
                answer = run_flow(user_input)
            
            # Rendering the output in markdown format
            st.subheader("Graphical Analysis")
            st.image("data/graph.jpg", caption="Generated Graph", use_container_width=True)
            st.subheader("Chatbot Response")
            st.markdown(answer['output'])
            
            # Displaying the graphical analysis
            
        else:
            st.warning("Please enter a message for the chatbot.")


elif page == "Engagement Analytics":
    # Engagement analytics page
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')


    st.header("Engagement Analytics")
    # Section 1: Engagement Overview

    st.write(' ')
    st.write(' ')
    st.markdown("#### 📊 Engagement Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Likes", int(df["Likes"].sum()))
    col2.metric("Total Comments", int(df["Comments"].sum()))
    col3.metric("Total Shares", int(df["Shares"].sum()))
    st.write(' ')
    st.write(' ')

    # Section 2: Engagement Breakdown
    st.markdown("#### 📈 Engagement Breakdown")
    fig = px.bar(
        df,
        x="Post_Timestamp",
        y=["Likes", "Comments", "Shares"],
        title="Engagement Breakdown",
        labels={"value": "Engagement Count", "variable": "Metric"},
        barmode="stack"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(' ')
    st.write(' ')

    # Section 3: Small Graphs in Two Columns
    st.markdown("#### 🔍 Insights from Engagement Metrics")
    st.write(' ')
    col1, col2 = st.columns(2)

    # Distribution of Metrics
    with col1:
        st.markdown("##### 📊 Distribution of Metrics")
        fig, ax = plt.subplots(figsize=(5, 4))
        for col in ["Likes", "Comments", "Shares"]:
            sns.histplot(df[col], kde=True, bins=20, ax=ax, label=col)
        ax.set_title("Distribution of Metrics")
        ax.set_xlabel("Count")
        ax.legend()
        st.pyplot(fig)

    # Box Plot for Outliers
    with col2:
        st.markdown("##### 📊 Outliers in Metrics")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df[["Likes", "Comments", "Shares"]], palette="Set2", ax=ax)
        ax.set_title("Outliers in Metrics")
        st.pyplot(fig)
    st.write(' ')
    st.write(' ')

    # Section 4: Engagement by Demographics
    st.markdown("#### 🌍 Audience Demographics Insights")
    st.write(' ')
    # ✅ Engagement by Gender and Age (Full Width)
    st.markdown("##### 📊 Engagement by Gender and Age (Expanded View)")
    df1 = df.drop(["Post_Timestamp"], axis=1)
    fig = px.bar(
        df1.groupby(["Audience_Gender", "Audience_Age"]).sum().reset_index(),
        x="Audience_Age",
        y=["Likes", "Comments", "Shares"],
        color="Audience_Gender",
        barmode="group",
        title="Engagement by Gender and Age"
    )
    st.plotly_chart(fig, use_container_width=True)  # Full width for better clarity
    st.write(' ')
    st.write(' ')


    # ✅ Placing Both World Maps Side by Side
    st.markdown("##### 🌍 Audience Engagement by Location (Global Comparison)")
    col1, col2 = st.columns(2)
    st.write(' ')
    st.write(' ')

    # Choropleth Map (Left Side)
    with col1:
        st.markdown("##### 🌍 Choropleth Map ")
        location_data = df.groupby("Audience_Location")[["Likes", "Comments", "Shares"]].sum().reset_index()
        location_data["Total_Engagement"] = location_data[["Likes", "Comments", "Shares"]].sum(axis=1)
        
        fig = px.choropleth(
            location_data,
            locations="Audience_Location",
            locationmode="country names",
            color="Total_Engagement",
            hover_name="Audience_Location",
            color_continuous_scale="Viridis",
            title="Audience Engagement by Location (Choropleth Map)",
            height=600,
        )
        fig.update_geos(showcoastlines=True, showland=True, landcolor="white", lakecolor="lightblue")
        st.plotly_chart(fig, use_container_width=True)

    # Bubble Map (Right Side)
    with col2:
        st.markdown("##### 📍 Bubble Map ")
        fig = px.scatter_geo(
            location_data,
            locations="Audience_Location",
            locationmode="country names",
            size="Total_Engagement",
            hover_name="Audience_Location",
            color="Total_Engagement",
            color_continuous_scale="Plasma",
            title="Audience Engagement by Location (Bubble Map)",
            height=600,
        )
        fig.update_geos(showland=True)
        st.plotly_chart(fig, use_container_width=True)
    st.write(' ')
    st.write(' ')


    # Section 5: Correlations and Sentiment
    st.markdown("#### 🔗 Correlations and Sentiment Insights")
    col1, col2 = st.columns(2)
    st.write(' ')

    # Engagement Rate vs Metrics
    with col1:
        st.markdown("##### 📈 Engagement Rate vs Metrics")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df, x="Engagement_Rate", y="Likes", label="Likes", color="blue", ax=ax)
        sns.scatterplot(data=df, x="Engagement_Rate", y="Comments", label="Comments", color="green", ax=ax)
        sns.scatterplot(data=df, x="Engagement_Rate", y="Shares", label="Shares", color="orange", ax=ax)
        ax.set_title("Engagement Rate vs Metrics")
        ax.set_xlabel("Engagement Rate")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

    # Engagement by Sentiment
    with col2:
        st.markdown("##### 📊 Engagement by Sentiment")
        df1 = df.drop(["Post_Timestamp"], axis=1)
        fig = px.bar(
            df1.groupby("Sentiment").sum().reset_index(),
            x="Sentiment",
            y=["Likes", "Comments", "Shares"],
            title="Engagement by Sentiment",
            labels={"value": "Engagement Count", "variable": "Metric"}
        )
        st.plotly_chart(fig, use_container_width=True)
    st.write(' ')
    st.write(' ')

    # Section 6: Word Cloud for Post Content
    st.markdown("#### 🖋 Text Analysis Insights")
    st.write(' ')
    st.markdown("##### 🌟 Most Common Words in Post Content")
    text = " ".join(df["Post_Content"].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # # Data Download Option
    # st.markdown("### 📥 Download Engagement Data")
    # csv = df.to_csv(index=False)
    # st.download_button(
    #     label="Download CSV",
    #     data=csv,
    #     file_name="engagement_data.csv",
    #     mime="text/csv",
    # )
# # Footer
# column
# st.markdown("""
# ---
# Developed with ❤ using Streamlit.
# """)