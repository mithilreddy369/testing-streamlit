import streamlit as st

# Title of the app
st.title('Basic Streamlit App')

# Create a text input for the user's name
name = st.text_input('Enter your name')

# Create a slider for the user's age
age = st.slider('Select your age', 0, 100, 25)

# Create a button to submit the information
if st.button('Submit'):
    # Display a personalized message
    st.write(f'Hello {name}, you are {age} years old!')
