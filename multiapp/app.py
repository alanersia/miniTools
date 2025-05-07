# File: app.py
import streamlit as st
import importlib
import pkgutil

# Base Page class
class Page:
    """Base class for all pages in the app."""
    def __init__(self, name: str):
        self.name = name

    def render(self):
        raise NotImplementedError("Each page must implement a render method.")

# App class to handle routing
class App:
    def __init__(self, pages: list[Page]):
        self.pages = {page.name: page for page in pages}

    def run(self):
        st.set_page_config(
            page_title="OOP Multipage App",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.sidebar.title("Navigasi")
        choice = st.sidebar.radio("Pilih Halaman", list(self.pages.keys()))
        self.pages[choice].render()

# Utility to dynamically load pages from 'pages' package
def load_pages():
    pages = []
    for finder, module_name, ispkg in pkgutil.iter_modules(['pages']):
        module = importlib.import_module(f'pages.{module_name}')
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, Page) and obj is not Page:
                pages.append(obj())
    return pages

if __name__ == '__main__':
    pages = load_pages()
    app = App(pages)
    app.run()