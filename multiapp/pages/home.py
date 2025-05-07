# File: pages/home.py
from app import Page
import streamlit as st

class HomePage(Page):
    def __init__(self):
        super().__init__("Home")

    def render(self):
        st.header("Selamat datang di Home Page")
        st.write("Pilih halaman lain melalui sidebar untuk memulai.")
