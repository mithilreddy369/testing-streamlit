import streamlit as st

def add_custom_css():
    st.markdown("""
    <style>
    /* General Body Styles */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f2f6;
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
        max-width: 900px; /* Ensure it does not become too wide */
        padding: 20px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
        height: 100%;
        box-sizing: border-box;
    }

    /* Header Styles */
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .header h1 {
        color: #0056b3;
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
        border-color: #0056b3;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
    }

    /* Button Styles */
    .btn-primary {
        background-color: #0056b3;
        border-color: #0056b3;
        color: #ffffff;
        font-weight: 600;
        border-radius: 4px;
        padding: 10px 20px;
    }
    .btn-primary:hover {
        background-color: #003d7a;
        border-color: #002a5e;
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
