import streamlit as st
from passlib.hash import bcrypt
from db import get_db_connection
from webapp import dashboard
import re
import time

def show_signup():
    user_details = {
        "mailid": None,
        "name": None,
        "age": None,
        "lang_pre": None,
        "pswd": None,
        "hashed_password": None
    }
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%?&])[A-Za-z\d@$!%?&]{8,}$'

    col_left, col_center, col_right = st.columns([1, 2, 1])  
    with col_center:
        st.title("📝 Sign Up")
        with st.form(key="user_registration_form", enter_to_submit=False):
            user_details["mailid"] = st.text_input("Email", placeholder="user@example.com")
            user_details["name"] = st.text_input("Name", placeholder="John Doe")
            user_details["age"] = st.selectbox("Age Group", ("18-25","26-35","36-45","46-55","56-65"))
            lang_options = ["English", "हिंदी"]
            user_details["lang_pre"] = st.radio("Language Preference", lang_options)
            user_details["pswd"] = st.text_input("Password", type="password")
            repswd = st.text_input("Re-enter Password", type="password")          
            submit = st.form_submit_button("Create Account")
            if submit:
                if not all([user_details["mailid"], user_details["name"], user_details["age"], user_details["lang_pre"], user_details["pswd"]]):
                    st.warning("Please fill all the fields")
                elif not re.match(email_regex, user_details["mailid"]):
                    st.warning("Please enter a valid email address")
                elif user_details["pswd"] != repswd:
                    st.warning("Passwords do not match")
                elif not re.match(password_regex, user_details["pswd"]):
                    st.warning("Password must be at least 8 characters long, include uppercase, lowercase, number, and special character")
                else:
                    user_details["hashed_password"] = bcrypt.hash(user_details["pswd"])
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT mailid FROM users WHERE mailid=%s", (user_details["mailid"],))
                        if cursor.fetchone():
                            st.warning("Email already registered")
                            cursor.close()
                            conn.close()
                            return
                        cursor.execute("INSERT INTO users (mailid, name, age_group, lang_pre, password_hash) VALUES (%s,%s,%s,%s,%s)",
                            (user_details["mailid"], user_details["name"], user_details["age"], user_details["lang_pre"], user_details["hashed_password"]))
                        conn.commit()
                        cursor.close()
                        conn.close()
                        st.success("Account created successfully! Redirecting to login page...")
                        time.sleep(2)
                        st.session_state.page = "signed_up"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        if st.button("Already have an account? Sign In"):
            st.session_state.page = "login"
            st.rerun()