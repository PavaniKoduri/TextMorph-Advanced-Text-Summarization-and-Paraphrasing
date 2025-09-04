import streamlit as st
import smtplib
from email.message import EmailMessage
import secrets
from datetime import datetime, timedelta
from passlib.hash import bcrypt
from db import get_db_connection

EMAIL_ADDRESS = st.secrets["email_address"]
EMAIL_PASSWORD = st.secrets["email_password"]
RESET_LINK_BASE = "http://localhost:8501"  

def send_reset_email(to_email, token):
    try:
        msg = EmailMessage()
        msg['Subject'] = "Password Reset Request"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        reset_link = f"{RESET_LINK_BASE}?reset_token={token}"
        msg.set_content(
            f"Hello,\n\n"
            f"You requested to reset your password. Click the link below to reset it:\n\n"
            f"{reset_link}\n\n"
            f"This link will expire in 30 minutes.\n\n"
            f"If you did not request a password reset, please ignore this email."
        )
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False
def forgot_password():
    st.title("🔑 Forgot Password")
    email_input = st.text_input("Enter your registered email")
    if st.button("Send Reset Link"):
        if not email_input:
            st.warning("Please enter your email")
            return
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT mailid FROM users WHERE mailid=%s", (email_input,))
            if not cursor.fetchone():
                st.error("Email not registered")
                cursor.close()
                conn.close()
                return
            token = secrets.token_urlsafe(16)
            expiry_time = datetime.now() + timedelta(minutes=30)
            cursor.execute("UPDATE users SET reset_token=%s, reset_token_expiry=%s WHERE mailid=%s",(token, expiry_time, email_input))
            conn.commit()
            cursor.close()
            conn.close()
            if send_reset_email(email_input, token):
                st.success("Reset link sent to your email. Please check your inbox.")
        except Exception as e:
            st.error(f"Database error: {e}")
def reset_password(token):
    if not token:
        st.error("Invalid or missing token")
        return
    st.session_state.reset_token = token
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mailid, reset_token_expiry FROM users WHERE reset_token=%s",(st.session_state.reset_token,))
        result = cursor.fetchone()
        if not result:
            st.error("Invalid or expired token")
            cursor.close()
            conn.close()
            return
        mailid, expiry = result
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")
        if datetime.now() > expiry:
            st.error("Token has expired. Request a new reset link.")
            cursor.execute("UPDATE users SET reset_token=NULL, reset_token_expiry=NULL WHERE mailid=%s",(mailid,))
            conn.commit()
            cursor.close()
            conn.close()
            st.session_state.pop("reset_token", None)
            return
        st.title("🔐 Reset Your Password")
        if "new_password" not in st.session_state:
            st.session_state.new_password = ""
        if "confirm_password" not in st.session_state:
            st.session_state.confirm_password = ""
        st.session_state.new_password = st.text_input("New Password", type="password", value=st.session_state.new_password)
        st.session_state.confirm_password = st.text_input("Confirm Password", type="password", value=st.session_state.confirm_password)
        if st.button("Update Password"):
            if not st.session_state.new_password or not st.session_state.confirm_password:
                st.warning("Fill both password fields")
                return
            if st.session_state.new_password != st.session_state.confirm_password:
                st.error("Passwords do not match")
                return
            hashed_pw = bcrypt.hash(st.session_state.new_password)
            cursor.execute("UPDATE users SET password_hash=%s, reset_token=NULL, reset_token_expiry=NULL WHERE mailid=%s",(hashed_pw, mailid))
            conn.commit()
            cursor.close()
            conn.close()
            for key in ["reset_token", "new_password", "confirm_password"]:
                st.session_state.pop(key, None)
            st.success("Password updated successfully. You can now close this page and login.")
    except Exception as e:
        st.error(f"Database error: {e}")
def main():
    token = st.query_params.get("reset_token", None)
    if token:
        reset_password(token)
    else:
        forgot_password()
if __name__ == "__main__":
    main()