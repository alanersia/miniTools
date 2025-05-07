# File: pages/calculator.py
from app import Page
import streamlit as st

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
        if operator == '+': return num1 + num2
        if operator == '-': return num1 - num2
        if operator == '*': return num1 * num2
        if operator == '/':
            if num2 == 0:
                return "Tidak bisa membagi dengan nol"
            return num1 / num2