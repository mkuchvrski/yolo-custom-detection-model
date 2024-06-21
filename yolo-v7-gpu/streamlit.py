import streamlit as st
from detect import detect

st.write("hello")

video = st.camera_input("video")

detect(video)