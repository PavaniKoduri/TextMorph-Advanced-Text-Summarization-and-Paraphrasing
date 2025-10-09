import streamlit as st
from webapp import profile, dashboard
import signup
import time
from db import get_db_connection
from datetime import datetime, timezone, timedelta
from passlib.hash import bcrypt
from jose import jwt
from forgotpassword import forgot_password, reset_password
from webapp import admin_dashboard




# ----------------------------
# 🔧 Sidebar Navigation Helper
# ----------------------------
def sidebar_navigation(current_tab=None):
    st.markdown("""
        <style>
        [data-testid="stSidebar"] .stRadio > div {
            font-size: 18px;
            font-weight: 600;
            padding: 10px 8px;
            line-height: 2.2;
        }
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #f0f0f0;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    menu_items = ["Dashboard", "Profile", "Logout"]
    if current_tab not in menu_items:
        current_tab = "Dashboard"
    choice = st.sidebar.radio("GO TO", menu_items, index=menu_items.index(current_tab))
    return choice


# ----------------------------
# 🔐 AUTH LOGIC STARTS HERE
# ----------------------------
reset_token = st.query_params.get("reset_token", None)

if reset_token:
    reset_password(reset_token)

else:
    # Initialize session states
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "admin_login_mode" not in st.session_state:
        st.session_state.admin_login_mode = False

    # Prevent admin from accessing signup directly
    if st.session_state.admin_login_mode and st.session_state.page == "signup":
        st.session_state.admin_login_mode = False

    # -----------------------------------------
    # LOGIN VIEW (USER / ADMIN)
    # -----------------------------------------
    if st.session_state.page in ["login", "signed_up"]:

        # USER LOGIN VIEW
        if not st.session_state.admin_login_mode:
            st.markdown("<h1 style='text-align:center; color:#4B9CD3;'>🔐 User Authentication</h1>", unsafe_allow_html=True)
            with st.container():
                st.write("Enter credentials to login or create a new account.")
                with st.form(key="user_auth_form", clear_on_submit=False):
                    form_values = {"mailid": None, "pswd": None}
                    form_values["mailid"] = st.text_input("Email", placeholder="user@example.com",
                                                          value=st.session_state.get("reset_email", ""))
                    if form_values["mailid"]:
                        st.session_state.reset_email = form_values["mailid"]

                    form_values["pswd"] = st.text_input("Password", type="password")

                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        signin = st.form_submit_button("Sign In")
                    with col2:
                        createacc = st.form_submit_button("Create Account")
                    with col3:
                        adminlogin = st.form_submit_button("Login as Admin")

                if st.button("Forgot Password?"):
                    if form_values["mailid"]:
                        st.session_state.reset_email = form_values["mailid"]
                    st.session_state.page = "forgot_password"
                    st.rerun()

            # USER LOGIN LOGIC
            if signin:
                if not all(form_values.values()):
                    st.warning("Please fill both email and password.")
                else:
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT password_hash, lang_pre, name FROM users WHERE mailid=%s",
                                       (form_values["mailid"],))
                        result = cursor.fetchone()
                        cursor.close()
                        conn.close()

                        if result:
                            stored_hash, lang_pref, name = result
                            if bcrypt.verify(form_values["pswd"], stored_hash):
                                st.session_state.page = "signed_in"
                                st.session_state.user_email = form_values["mailid"]
                                st.session_state.user_name = name
                                st.session_state.lang_pref = lang_pref
                                payload = {
                                    "mailid": form_values["mailid"],
                                    "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
                                }
                                st.session_state.jwt_token = jwt.encode(payload, st.secrets["jwt_secret"], algorithm="HS256")
                                st.success(f"Welcome, {name}!")
                                st.rerun()
                            else:
                                st.error("Incorrect password")
                        else:
                            st.error("Email not registered")
                    except Exception as e:
                        st.error(f"Database error: {e}")

            if createacc:
                st.session_state.page = "signup"
                st.rerun()

            if adminlogin:
                st.session_state.admin_login_mode = True
                st.rerun()

        # ADMIN LOGIN VIEW
        else:
            st.markdown("<h1 style='text-align:center; color:#D9534F;'>🛠️ Admin Login</h1>", unsafe_allow_html=True)
            st.write("Admins only. Please enter your credentials.")

            admin_email = st.text_input("Admin Email", placeholder="admin@textmorph.com")
            admin_password = st.text_input("Admin Password", type="password")

            col1, col2 = st.columns([1, 1])
            with col1:
                admin_signin = st.button("Sign In as Admin")
            with col2:
                back_to_user = st.button("⬅️ Back to User Login")

            if admin_signin:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT password_hash, name FROM admins WHERE email=%s", (admin_email,))
                    result = cursor.fetchone()
                    cursor.close()
                    conn.close()

                    if result:
                        stored_hash, admin_name = result
                        if bcrypt.verify(admin_password, stored_hash):
                            st.success(f"✅ Welcome back, {admin_name}!")
                            st.session_state.page = "admin_dashboard"
                            st.session_state.admin_email = admin_email
                            st.session_state.admin_name = admin_name
                            st.session_state.admin_login_mode = False
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Incorrect password.")
                    else:
                        st.error("❌ Admin account not found.")
                except Exception as e:
                    st.error(f"Database error: {e}")


            if back_to_user:
                st.session_state.admin_login_mode = False
                st.rerun()

    # -----------------------------------------
    # SIGNUP PAGE
    # -----------------------------------------
    elif st.session_state.page == "signup":
        signup.show_signup()

    # -----------------------------------------
    # ADMIN DASHBOARD PAGE
    # -----------------------------------------
    elif st.session_state.page == "admin_dashboard":
        admin_dashboard.show_admin_dashboard()

        if st.button("🔒 Logout as Admin"):
            for key in ["admin_email", "admin_login_mode"]:
                st.session_state.pop(key, None)
            st.session_state.page = "login"
            st.info("Logging out as admin...")
            time.sleep(1)
            st.rerun()

    # -----------------------------------------
    # USER SIGNED-IN DASHBOARD
    # -----------------------------------------
    elif st.session_state.page == "signed_in":
        menu = sidebar_navigation(st.session_state.get("current_tab", "Dashboard"))
        st.session_state.current_tab = menu
        if menu == "Dashboard":
            dashboard.show_dashboard()
        elif menu == "Profile":
            profile.show_profile()
        elif menu == "Logout":
            for key in ["user_email", "user_name", "lang_pref", "jwt_token", "current_tab"]:
                st.session_state.pop(key, None)
            st.session_state.page = "login"
            st.info("Logging out...")
            time.sleep(1)
            st.rerun()

    # -----------------------------------------
    # FORGOT PASSWORD PAGE
    # -----------------------------------------
    elif st.session_state.page == "forgot_password":
        forgot_password()

    # -----------------------------------------
    # FALLBACK (safety net)
    # -----------------------------------------
    else:
        st.error("⚠️ Unknown page state. Resetting session.")
        st.session_state.page = "login"
        st.rerun()
