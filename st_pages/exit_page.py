"""
Exit Page Module
"""
import streamlit as st
import os

def show():
    st.title("🚪 Exit Application")
    
    # Display confirmation
    st.warning("Are you sure you want to exit?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, Exit"):
            # Clear session state
            st.session_state.clear()
            # Display goodbye message
            st.success("Thank you for using the application!")
            # Add delay before closing
            st.write("Closing application...")
            # Stop the Streamlit server
            os._exit(0)
            
    with col2:
        if st.button("❌ No, Go Back"):
            # Rerun to go back to previous page
            st.rerun()