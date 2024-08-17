import streamlit as st

def add_custom_css():
    st.markdown("""
    <style>
    /* General Body Styles */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #121212;
        color: #e0e0e0;
        margin: 0;
        padding: 0;
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Container Styles */
    .container {
        width: 80%;
        max-width: 1000px; /* Adjusted for 3 columns */
        padding: 20px;
        background-color: #1e1e1e;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        height: 100%;
        box-sizing: border-box;
    }

    /* Header Styles */
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    .header h1 {
        color: #f5f5f5;
        font-size: 2.5em;
        font-weight: 700;
    }

    /* Form Styles */
    .form-group {
        margin-bottom: 20px;
        color: #e0e0e0;
    }
    .form-group label {
        display: block;
        font-weight: 600;
        color: #e0e0e0;
    }
    .form-group input, .form-group select {
        width: 100%;
        padding: 10px;
        border: 1px solid #333;
        border-radius: 4px;
        background-color: #333;
        color: #e0e0e0;
    }
    .form-group input:focus, .form-group select:focus {
        border-color: #00bcd4;
        box-shadow: 0 0 0 0.2rem rgba(0,188,212,0.25);
    }

    /* Button Styles */
    .btn-primary {
        background-color: #00bcd4;
        border-color: #00bcd4;
        color: #ffffff;
        font-weight: 600;
        border-radius: 4px;
        padding: 10px 20px;
    }
    .btn-primary:hover {
        background-color: #0097a7;
        border-color: #00838f;
    }

    /* Alert Styles */
    .alert {
        border-radius: 4px;
        padding: 15px;
        font-size: 1.1em;
    }
    .alert-primary {
        background-color: #263238;
        color: #b0bec5;
    }
    .alert-heading {
        font-size: 1.2em;
        font-weight: 600;
    }

    /* Grid Layout for Form */
    .form-row {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }
    .form-column {
        flex: 1;
        min-width: 200px; /* Adjust based on needs */
    }
    </style>
    """, unsafe_allow_html=True)
