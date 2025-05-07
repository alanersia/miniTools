import streamlit as st

class Page:
    """
    Base class for all pages in the app.
    """
    def __init__(self, name: str):
        self.name = name

    def render(self):
        """Override this method to render the page content."""
        raise NotImplementedError("Each page must implement a render method.")

class HomePage(Page):
    def __init__(self):
        super().__init__("Home")

    def render(self):
        st.header("Selamat datang di Home Page")
        st.write("Pilih halaman lain melalui sidebar untuk memulai.")

class CalculatorPage(Page):
    def __init__(self):
        super().__init__("Calculator")
        self.operators = ['+', '-', '*', '/']

    def render(self):
        st.header("Simple Calculator")
        num1 = st.number_input("Number 1", value=0.0)
        num2 = st.number_input("Number 2", value=0.0)
        op = st.selectbox("Operator", self.operators)
        result = self.calculate(num1, num2, op)
        st.subheader("Hasil")
        st.write(f"{num1} {op} {num2} = {result}")

    def calculate(self, num1, num2, operator):
        if operator == '+':
            return num1 + num2
        if operator == '-':
            return num1 - num2
        if operator == '*':
            return num1 * num2
        if operator == '/':
            if num2 == 0:
                return "Tidak bisa membagi dengan nol"
            return num1 / num2

class AboutPage(Page):
    def __init__(self):
        super().__init__("About")

    def render(self):
        st.header("Tentang Aplikasi")
        st.write("Aplikasi ini dibuat dengan konsep OOP dan Streamlit multipage routing manual.")

class App:
    """
    Main application class to handle routing between pages.
    """
    def __init__(self, pages: list[Page]):
        self.pages = {page.name: page for page in pages}

    def run(self):
        st.sidebar.title("Navigasi")
        choice = st.sidebar.radio("Pilih Halaman", list(self.pages.keys()))
        page = self.pages[choice]
        page.render()

if __name__ == '__main__':
    # Daftar halaman
    pages = [HomePage(), CalculatorPage(), AboutPage()]
    app = App(pages)
    app.run()
