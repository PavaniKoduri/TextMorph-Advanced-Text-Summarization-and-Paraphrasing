import streamlit as st
from db import get_db_connection
from passlib.hash import bcrypt
import re

password_regex = r'^(?=.[a-z])(?=.[A-Z])(?=.\d)(?=.[@$!%?&])[A-Za-z\d@$!%?&]{8,}$'

def show_profile():
    st.title("👤 My Profile")
    email = st.session_state.get("user_email")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, age_group, lang_pre FROM users WHERE mailid=%s", (email,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if not result:
            st.error("User not found")
            return
        name, age_group, lang_pre = result
        with st.container():
            st.markdown(
            f"""
            <div style='background-color:#EAF2F8; padding:20px; border-radius:10px;'>
                <p style='font-weight:bold; color:#1B2631;'>📧 Email:</p> <p style='color:#2E86C1;'>{email}</p>
                <p style='font-weight:bold; color:#1B2631;'>👤 Name:</p> <p style='color:#2E86C1;'>{name}</p>
                <p style='font-weight:bold; color:#1B2631;'>🗓 Age Group:</p> <p style='color:#2E86C1;'>{age_group}</p>
                <p style='font-weight:bold; color:#1B2631;'>🗣 Language Preference:</p> <p style='color:#2E86C1;'>{lang_pre}</p>
            </div>
            """,unsafe_allow_html=True
            )
        if st.checkbox("Edit Profile"):
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                with st.form(key="edit_profile_form", enter_to_submit=False):
                    updated_name = st.text_input("Name", value=name)
                    age_options = ["18-25","26-35","36-45","46-55","56-65"]
                    updated_age = st.selectbox("Age Group", age_options, index=age_options.index(age_group))
                    lang_options = ["English","हिंदी"]
                    updated_lang = st.selectbox("Language Preference", lang_options, index=lang_options.index(lang_pre))
                    change_password = st.checkbox("Change Password?")
                    new_password = st.text_input("New Password", type="password") if change_password else ""
                    confirm_password = st.text_input("Confirm New Password", type="password") if change_password else ""
                    submit = st.form_submit_button("Update Profile")
                    if submit:
                        if not updated_name or not updated_age or not updated_lang:
                            st.warning("Please fill all fields")
                            return
                        hashed_password = None
                        if change_password:
                            if not new_password or not confirm_password:
                                st.warning("Please fill password fields")
                                return
                            elif new_password != confirm_password:
                                st.warning("Passwords do not match")
                                return
                            elif not re.match(password_regex, new_password):
                                st.warning("Password must be at least 8 characters, include uppercase, lowercase, number, and special character")
                                return
                            hashed_password = bcrypt.hash(new_password)
                        try:
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            if hashed_password:
                                cursor.execute("UPDATE users SET name=%s, age_group=%s, lang_pre=%s, password_hash=%s WHERE mailid=%s",
                                    (updated_name, updated_age, updated_lang, hashed_password, email))
                            else:
                                cursor.execute("UPDATE users SET name=%s, age_group=%s, lang_pre=%s WHERE mailid=%s",
                                    (updated_name, updated_age, updated_lang, email))
                            conn.commit()
                            cursor.close()
                            conn.close()
                            st.session_state.user_name = updated_name
                            st.session_state.lang_pref = updated_lang
                            st.success("Profile updated successfully")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Database error: {e}")
    except Exception as e:
        st.error(f"Database error: {e}")