import streamlit as st
import requests
import textstat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
API_URL = "http://127.0.0.1:8000"
def get_color(score_name, value):
    if score_name == "Flesch Reading Ease":
        if value >= 60:
            return "#27AE60"  
        elif value >= 30:
            return "#F1C40F"  
        else:
            return "#E74C3C"  
    else:
        if value <= 6:
            return "#27AE60"
        elif value <= 12:
            return "#F1C40F"
        else:
            return "#E74C3C"
def show_dashboard():
    st.title("📊 Dashboard")
    user_email = st.session_state.get("user_email")
    if not user_email:
        st.error("You must be logged in to use the dashboard")
        return
    st.subheader("Upload a Text File for Readability Analysis")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.upload_success = False
        if not st.session_state.get("upload_success", False):
            if st.button("Upload"):
                try:
                    files = {"uploaded_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"user_email": user_email}
                    response = requests.post(f"{API_URL}/upload", data=data, files=files, timeout=10)
                    if response.status_code == 200:
                        st.success(response.json().get("message", "File uploaded successfully"))
                        st.session_state.upload_success = True
                        st.session_state.last_uploaded_file = uploaded_file.name
                    else:
                        st.error(f"Upload failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the backend: {e}")
        if st.session_state.get("upload_success", False):
            try:
                content = uploaded_file.getvalue().decode("utf-8")
                scores = {"Flesch Reading Ease": textstat.flesch_reading_ease(content),"Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(content),
                    "SMOG Index": textstat.smog_index(content),"Automated Readability Index": textstat.automated_readability_index(content)}
                st.markdown("<h3 style='color:#34495E'>Readability Analysis:</h3>", unsafe_allow_html=True)            
                for k, v in scores.items():
                    st.markdown(f"<span style='color:#1F618D; font-weight:bold'>{k}:</span> <span style='color:#2C3E50'>{v:.2f}</span>", unsafe_allow_html=True)
                colors = [get_color(name, value) for name, value in scores.items()]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(list(scores.keys()), list(scores.values()), color=colors)
                ax.set_xlabel("Score")
                ax.set_title("Readability Scores", fontsize=14, color="#2C3E50")
                green_patch = mpatches.Patch(color='#27AE60', label='Easy')
                yellow_patch = mpatches.Patch(color='#F1C40F', label='Moderate')
                red_patch = mpatches.Patch(color='#E74C3C', label='Hard')
                ax.legend(handles=[green_patch, yellow_patch, red_patch], loc='lower right')
                st.pyplot(fig)
            except Exception:
                st.error("Failed to read file content for analysis")