import streamlit as st

def add_custom_css():
    st.markdown("""
    <style>
    /* General Body Styles */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f8f9fa;
    }

    /* Container Styles */
    .container {
        width: 80%;
        margin: auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }

    /* Header Styles */
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    .header h1 {
        color: #007bff;
        font-size: 2.5em;
        font-weight: 700;
    }

    /* Form Styles */
    .form-group {
        margin-bottom: 20px;
    }
    .form-group label {
        display: block;
        font-weight: 600;
        color: #333;
    }
    .form-group input, .form-group select {
        width: 100%;
        padding: 10px;
        border: 1px solid #ced4da;
        border-radius: 4px;
    }
    .form-group input:focus, .form-group select:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
    }

    /* Button Styles */
    .btn-primary {
        background-color: #007bff;
        border-color: #007bff;
        color: #ffffff;
        font-weight: 600;
    }
    .btn-primary:hover {
        background-color: #0056b3;
        border-color: #004085;
    }

    /* Alert Styles */
    .alert {
        border-radius: 4px;
        padding: 15px;
        font-size: 1.1em;
    }
    .alert-primary {
        background-color: #cce5ff;
        color: #004085;
    }
    .alert-heading {
        font-size: 1.2em;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Call this function in your main app file to apply the styles
