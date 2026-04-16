import sys
import os

# Add the current directory to sys.path to allow imports from app.py
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from streamlit_serverless import streamlit_serverless
import app

def handler(event, context):
    return streamlit_serverless(app.run)
