import streamlit as st
import requests
import textstat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os
from datetime import datetime
import pdfplumber
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
import PyPDF2
import docx
import plotly.express as px
from rouge_score import rouge_scorer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

API_URL = "http://127.0.0.1:8000"

LOCAL_SAVE_DIR = "manual_inputs"
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

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



@st.cache_resource
def load_summarizer(model_name):
    return pipeline("summarization", model=model_name)

@st.cache_resource
def load_paraphraser(model_name):
    return pipeline("text2text-generation", model=model_name)

SUMMARIZATION_MODELS = {
    "BART (facebook/bart-large-cnn)": "facebook/bart-large-cnn",
    "T5 (google/flan-t5-large)": "google/flan-t5-large",
    "PEGASUS (google/pegasus-xsum)": "google/pegasus-xsum"
}


def analyze_and_show(content):
    scores = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(content),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(content),
        "SMOG Index": textstat.smog_index(content),
        "Automated Readability Index": textstat.automated_readability_index(content)
    }

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
    st.markdown("""
        📖 **What this shows:**  
        The bars below represent different **readability scores** for your text.  

        - **Flesch Reading Ease** → Higher = easier to read  
        - **Flesch-Kincaid / SMOG / ARI** → Approximate grade level required to understand  
        - Green = Easy, Yellow = Moderate, Red = Hard
    """)
    st.pyplot(fig)

def store_summary_db(user_email, original_text, summary_text, model_used, summary_length, reference_summary="", rouge_scores={}):
    payload = {
        "user_email": user_email,
        "original_text": original_text,
        "summary_text": summary_text,
        "model_used": model_used,
        "summary_length": summary_length,
        "reference_summary": reference_summary,
        "rouge_scores": rouge_scores
    }
    try:
        resp = requests.post(f"{API_URL}/store_evaluation", json=payload, timeout=10)
        if resp.status_code == 200:
            st.success(f"Summary ({summary_length}) stored successfully in DB")
        else:
            st.error(f"Failed to store summary: {resp.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to backend: {e}")


def show_dashboard():
    st.title("📊TextMorph's Dashboard")
    user_email = st.session_state.get("user_email")
    if not user_email:
        st.error("You must be logged in to use the dashboard")
        return

    
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(["Readability Analysis", "Summarize", "Paraphrase","History"])

    with main_tab1:
        st.subheader("Readablility Scores")
        tab1, tab2 = st.tabs(["Manual Input", "Upload File"])
        with tab1:
            st.markdown("Enter Text Manually")
            manual_text = st.text_area("Enter your text here", height=200)

            if st.button("Save & Analyze Manual Text"):
                if manual_text.strip() == "":
                    st.warning("Please enter some text before saving.")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_{user_email}_{timestamp}.txt"
                    filepath = os.path.join(LOCAL_SAVE_DIR, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(manual_text)
                    st.success(f"Manual text uploaded")
                    analyze_and_show(manual_text)

        
        with tab2:
            st.markdown("Upload a Text File for Readability Analysis")
            uploaded_file = st.file_uploader("Choose a .txt or a .pdf file", type=["txt","pdf"])
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
                        content = None
                        if uploaded_file.type == "text/plain":  # TXT file
                            content = uploaded_file.getvalue().decode("utf-8")
                        elif uploaded_file.type == "application/pdf":  # PDF file
                            with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                                content = "\n".join(page.extract_text() or "" for page in pdf.pages)

                        if content and content.strip():
                            analyze_and_show(content)
                        else:
                            st.error("No readable text found in this file.")
                    except Exception as e:
                        st.error(f"Failed to read file content for analysis: {e}")

    
    with main_tab2:

        st.subheader("Summarize Text or PDF")

        col1, col2 = st.columns(2)
        with col1:
            input_type = st.radio("Choose input type:", ["Plain Text","Text File","PDF File"])
            model_choice = st.selectbox(
                "Select Model",
                options=["pegasus", "bart", "flan-t5"],
                index=0
            )
            summary_length = st.selectbox(
                "Summary Length",
                options=["short", "medium", "long"],
                index=1
            )
        with col2:
            st.write("Instructions:")
            st.markdown("- Paste text or upload a .txt or .pdf file.\n- Choose a model and summary length.\n- Click 'Generate Summary'.")

        uploaded_file = None
        text_input=""
        if input_type == "Plain Text":
            text_input = st.text_area("Paste your text here", height=200)
        elif input_type == "Text File":
            uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"], key="file_uploader_txt")
        elif input_type == "PDF File":
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="file_uploader_pdf")


        reference_input = st.text_area(
            "Reference Summary (optional for ROUGE evaluation)",
            height=150,
            help="Paste a human-written summary here to compute ROUGE metrics."
        )

        if 'last_summary' not in st.session_state:
            st.session_state['last_summary'] = None
            st.session_state['last_model'] = None
            st.session_state['last_length'] = None

        if st.button("Generate Summary"):
            
            try:
                
                original_text_display = ""

                if input_type == "Plain Text":
                    original_text_display = text_input.strip()

                elif input_type == "Text File" and uploaded_file:
                    original_text_display = uploaded_file.getvalue().decode("utf-8").strip()

                elif input_type == "PDF File" and uploaded_file:
                    with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                        original_text_display = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()


                if not original_text_display:
                    st.error("No valid text found to summarize.")
                else:
                    with st.spinner("Generating summary..."):
                        model_map = {
                            "pegasus": "google/pegasus-xsum",
                            "bart": "facebook/bart-large-cnn",
                            "flan-t5": "google/flan-t5-large"
                        }

                        summarizer = load_summarizer(model_map[model_choice])

                        length_map = {"short": (20, 60), "medium": (60, 120), "long": (120, 200)}
                        min_len, max_len = length_map[summary_length]

                        result = summarizer(original_text_display, min_length=min_len, max_length=max_len, do_sample=False)
                        summary_text_display = result[0]['summary_text'].strip()

                        wc_orig = len(original_text_display.split())
                        wc_sum = len(summary_text_display.split())
                        compression = (1 - (wc_sum / wc_orig)) * 100 if wc_orig > 0 else 0

                        col_orig, col_sum = st.columns(2)
                        with col_orig:
                            st.markdown("#### Original Text")
                            st.caption(f"{wc_orig} words")
                            st.text_area("Original", value=original_text_display[:15000], height=260)
                        with col_sum:
                            st.markdown("#### Summary")
                            st.caption(f"{wc_sum} words")
                            st.text_area("Summary", value=summary_text_display, height=260)
                            st.markdown(f"**Compression:** {compression:.0f}%")

                        
                        st.session_state['last_summary'] = summary_text_display
                        st.session_state['last_model'] = model_choice
                        st.session_state['last_length'] = summary_length

                        
                        scores = {}
                        if reference_input.strip():
                            rouge = evaluate.load("rouge")
                            scores = rouge.compute(predictions=[summary_text_display], references=[reference_input.strip()])

                            st.markdown("### ROUGE Evaluation")
                            st.json(scores)

                            
                            df_scores = pd.DataFrame(list(scores.items()), columns=["Metric", "Score"])
                            st.dataframe(df_scores)

                           
                            csv_bytes = df_scores.to_csv(index=False).encode()
                            st.download_button("Download ROUGE CSV", data=csv_bytes, file_name="rouge_scores.csv")

                            
                            st.markdown("""
                                🔍 **What this shows:**  
                                The bars below compare your **generated summary** with the **reference summary**.  

                                - **ROUGE-1** → word overlap  
                                - **ROUGE-2** → two-word phrase overlap  
                                - **ROUGE-L** → longest common sequence  

                                Higher values = summary is closer in meaning to the reference.
                            """)
                            fig, ax = plt.subplots()
                            ax.bar(df_scores['Metric'], df_scores['Score'], color="#3b82f6")
                            ax.set_ylim(0, 1)
                            ax.set_ylabel("Score")
                            ax.set_title("ROUGE Scores")
                            st.pyplot(fig)
                        store_summary_db(
                            user_email=user_email,
                            original_text=original_text_display,
                            summary_text=summary_text_display,
                            model_used=model_choice,
                            summary_length=summary_length,
                            reference_summary=reference_input.strip(),
                            rouge_scores=scores if reference_input.strip() else {}
                        )
            except Exception as e:
                st.error(f"An error occurred: {e}")

    
    def store_paraphrase_db(user_email, original_text, paraphrased_results, model_used, creativity, complexity_level, rouge_scores, readability_scores):
        payload = {
            "user_email": user_email,
            "original_text": original_text,
            "paraphrased_options": paraphrased_results,
            "model_used": model_used,
            "creativity": creativity,
            "complexity_level": complexity_level,
            "rouge_scores": rouge_scores,
            "readability_scores": readability_scores
        }
        try:
            resp = requests.post(f"{API_URL}/store_paraphrase", json=payload, timeout=10)
            if resp.status_code == 200:
                st.success("All paraphrases stored successfully in DB")
            else:
                st.error(f"Failed to store paraphrases: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to backend: {e}")

    def paraphrasing_ui(user_email):
        st.subheader("Paraphrasing & Analysis")
        input_method = st.radio("Choose input method:", ["Text Input", "File Upload"], horizontal=True)
        original_text = ""

        if input_method == "Text Input":
            original_text = st.text_area("Enter text to paraphrase", height=200)
        else:
            uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt","pdf","docx"])
            if uploaded_file:
                file_bytes = uploaded_file.getvalue()
                if uploaded_file.type == "application/pdf":
                    import PyPDF2, io
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                    original_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    import docx, io
                    doc = docx.Document(io.BytesIO(file_bytes))
                    original_text = "\n".join(para.text for para in doc.paragraphs)
                else:
                    original_text = file_bytes.decode("utf-8")

        col1, col2 = st.columns(2)
        with col1:
            creativity = st.slider("Creativity", 0.5, 1.5, 1.0, 0.1)
        with col2:
            complexity_level = st.selectbox("Complexity Level", ["Beginner", "Intermediate", "Advanced"])

        paraphrase_models = {
            "T5 Paraphraser (Humarin)": "humarin/chatgpt_paraphraser_on_T5_base",
            "Pegasus (Google)": "tuner007/pegasus_paraphrase",
            "BART (Facebook)": "eugenesiow/bart-paraphrase"
        }
        selected_model = st.selectbox("Select Model", list(paraphrase_models.keys()))

        complexity_map = {"Beginner": 128, "Intermediate": 256, "Advanced": 512}
        max_len = complexity_map[complexity_level]

        complexity_prompt_map = {
            "Beginner": "Paraphrase the following text in simple and clear language suitable for beginners:",
            "Intermediate": "Paraphrase the following text with moderate complexity suitable for intermediate readers:",
            "Advanced": "Paraphrase the following text with advanced vocabulary and sentence structure suitable for expert readers:"
        }
        prompt_text = complexity_prompt_map[complexity_level] + "\n" + original_text

        if st.button("Generate & Analyze", type="primary") and original_text.strip():
            para_pipe = load_paraphraser(paraphrase_models[selected_model])
            outputs = para_pipe(
                prompt_text,
                num_return_sequences=3,
                num_beams=5,
                temperature=creativity,
                max_length=max_len,
                truncation=True
            )
            paraphrased_results = [o["generated_text"] for o in outputs]

            st.subheader("Paraphrased Options")
            for i, txt in enumerate(paraphrased_results, 1):
                st.write(f"**Option {i}:**")
                st.info(txt)

            import textstat, pandas as pd, plotly.express as px
            complexity_data = [{"Source": "Original", "Score": textstat.flesch_reading_ease(original_text)}]
            for i, txt in enumerate(paraphrased_results, 1):
                complexity_data.append({"Source": f"Option {i}", "Score": textstat.flesch_reading_ease(txt)})
            df_complexity = pd.DataFrame(complexity_data)

            st.subheader("Readability Analysis")
            fig = px.bar(df_complexity, x="Source", y="Score",
                        color="Source", title="Flesch Reading Ease", template="plotly_white")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
            scores_data = []
            for i, txt in enumerate(paraphrased_results, 1):
                scores = scorer.score(original_text, txt)
                scores_data.append({
                    "Option": f"Option {i}",
                    "ROUGE-1": scores['rouge1'].fmeasure,
                    "ROUGE-2": scores['rouge2'].fmeasure,
                    "ROUGE-L": scores['rougeL'].fmeasure
                })
            df_scores = pd.DataFrame(scores_data)

            st.subheader("ROUGE Comparison")
            fig2 = px.bar(
                df_scores.melt(id_vars="Option", var_name="Metric", value_name="Score"),
                x="Option", y="Score", color="Metric", barmode="group",
                title="ROUGE F1-Scores vs Original", template="plotly_white"
            )
            st.plotly_chart(fig2, use_container_width=True)

            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sid = SentimentIntensityAnalyzer()

            sentiment_orig = sid.polarity_scores(original_text)
            st.subheader("Sentiment Analysis (Original Text)")
            pie_data_orig = {k: v for k, v in sentiment_orig.items() if k != 'compound'}
            fig3 = px.pie(names=list(pie_data_orig.keys()), values=list(pie_data_orig.values()),
                        title="Original Text Sentiment", template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)
            st.json(sentiment_orig)

            sentiment_list = [sid.polarity_scores(txt) for txt in paraphrased_results]
            avg_sentiment = {k: sum(d[k] for d in sentiment_list)/len(sentiment_list) for k in sentiment_list[0] if k != 'compound'}
            st.subheader("Average Sentiment (Paraphrased Texts)")
            fig4 = px.pie(names=list(avg_sentiment.keys()), values=list(avg_sentiment.values()),
                        title="Paraphrases Average Sentiment", template="plotly_white")
            st.plotly_chart(fig4, use_container_width=True)
            st.json(avg_sentiment)

            combined_text = "Original:\n" + original_text + "\n\n"
            for i, txt in enumerate(paraphrased_results, 1):
                combined_text += f"Option {i}:\n{txt}\n\n"
            
            store_paraphrase_db(
                user_email=user_email,
                original_text=original_text,
                paraphrased_results=paraphrased_results,
                model_used=selected_model,
                creativity=creativity,
                complexity_level=complexity_level,
                rouge_scores=df_scores.to_dict(orient="records"),
                readability_scores=df_complexity.to_dict(orient="records")
            )


            st.download_button("📥 Download Paraphrases", data=combined_text.encode("utf-8"),
                            file_name="paraphrased_results.txt", mime="text/plain")

    with main_tab3:
        paraphrasing_ui(user_email)

    def show_history(user_email):
        st.subheader("Your History")

        history_tab1, history_tab2, history_tab3 = st.tabs(["Summaries", "Paraphrases", "Readability Analysis"])

        with history_tab1:
            st.markdown("### Summaries History")
            try:
                resp = requests.get(f"{API_URL}/history/summaries/{user_email}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df[["original_text", "summary_text", "model_used", "summary_length", "created_at"]])
                        csv_bytes = df.to_csv(index=False).encode()
                        st.download_button("Download CSV", data=csv_bytes, file_name="summaries_history.csv")
                    else:
                        st.info("No summaries found.")
                else:
                    st.error(f"Failed to fetch summaries: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")

        with history_tab2:
            st.markdown("### Paraphrases History")
            try:
                resp = requests.get(f"{API_URL}/history/paraphrases/{user_email}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df[["original_text", "paraphrased_options", "model_used", "complexity_level", "created_at"]])
                        csv_bytes = df.to_csv(index=False).encode()
                        st.download_button("Download CSV", data=csv_bytes, file_name="paraphrases_history.csv")
                    else:
                        st.info("No paraphrases found.")
                else:
                    st.error(f"Failed to fetch paraphrases: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")

        # 3️⃣ Readability Analysis History

        with history_tab3:
            st.markdown("### Readability Analysis History")
            try:
                resp = requests.get(f"{API_URL}/history/uploaded_files/{user_email}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df[["filename", "filetype", "filesize", "uploaded_at"]])

                        # Use expanders for each file
                        for i, row in df.iterrows():
                            with st.expander(f"View Content: {row['filename']}"):
                                try:
                                    content_resp = requests.get(f"{API_URL}/history/uploaded_files/content/{row['id']}", timeout=10)
                                    if content_resp.status_code == 200:
                                        file_content = content_resp.json().get("content", "")
                                        # Add unique key here
                                        st.text_area("File Content", value=file_content, height=300, key=f"file_content_{row['id']}")
                                    else:
                                        st.error(f"Failed to fetch content: {content_resp.text}")
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Could not connect to backend: {e}")

                        csv_bytes = df.to_csv(index=False).encode()
                        st.download_button("Download CSV", data=csv_bytes, file_name="readability_history.csv")
                    else:
                        st.info("No readability analysis found.")
                else:
                    st.error(f"Failed to fetch uploaded files: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")


    with main_tab4:
        show_history(user_email)

