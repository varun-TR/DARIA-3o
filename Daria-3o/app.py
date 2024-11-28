import streamlit as st
from test2 import main as main_page   
from chat2 import main as chat_page

# Add background image
background_image_url = "https://images.unsplash.com/photo-1421941629638-ded5fddb2300?q=80&w=2048&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
background_css = f"""
<style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Initialize session state for screen toggle
if "current_screen" not in st.session_state:
    st.session_state["current_screen"] = "main"  # Default to the main page

# Sidebar for navigation
st.sidebar.title("Navigation")
if st.session_state["current_screen"] == "main":
    st.sidebar.button("Chat with llama", on_click=lambda: st.session_state.update({"current_screen": "chat"}))
elif st.session_state["current_screen"] == "chat":
    st.sidebar.button("Back to Main", on_click=lambda: st.session_state.update({"current_screen": "main"}))

# Render appropriate page based on current screen
if st.session_state["current_screen"] == "main":
    main_page()  # Call the main function from test2.py
elif st.session_state["current_screen"] == "chat":
    chat_page()  # Call the main function from chat2.py
