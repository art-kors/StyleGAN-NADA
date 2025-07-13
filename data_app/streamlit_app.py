import streamlit as st

import os

print(os.getcwd())

st.title("An inference showcase of my reimplementation")

video_file = open("./data_app/inference showcase.mp4", "rb")
video_bytes = video_file.read()

st.video(video_bytes)