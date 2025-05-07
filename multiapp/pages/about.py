# File: pages/about.py
from app import Page
import streamlit as st

class AboutPage(Page):
    def __init__(self):
        super().__init__("About")

    def render(self):
        st.header("Tentang Aplikasi")
        st.write("Aplikasi ini dibuat dengan konsep OOP dan multipage routing dari beberapa file.")
