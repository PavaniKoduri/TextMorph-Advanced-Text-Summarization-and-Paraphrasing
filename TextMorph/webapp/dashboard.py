import math
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import json
import streamlit as st
import requests
import textstat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os
import re
from datetime import datetime
import pdfplumber
import transformers
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
import PyPDF2
import docx
import plotly.express as px
from rouge_score import rouge_scorer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import datasets
import torch
from googletrans import Translator
translator = Translator()
import time
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


from sentence_transformers import SentenceTransformer, util
if "sim_model" not in st.session_state:
    st.session_state["sim_model"] = SentenceTransformer("all-MiniLM-L6-v2")


def safe_bleu(preds, refs):
    """
    Always returns a dict with 'bleu' and 'precisions' keys.
    """
    preds = [p for p in preds if p and p.strip()]
    refs = [[r] for r in refs if r and r.strip()]
    if not preds or not refs:
        return {"bleu": 0.0, "precisions": [0.0, 0.0, 0.0, 0.0]}
    try:
        res = bleu.compute(predictions=preds, references=refs)
        return {
            "bleu": float(res.get("bleu", 0.0)),
            "precisions": list(res.get("precisions", [0.0, 0.0, 0.0, 0.0]))
        }
    except Exception:
        return {"bleu": 0.0, "precisions": [0.0, 0.0, 0.0, 0.0]}

def safe_translate(text, dest_lang):
    """Translate text with error handling and tiny rate-limit sleep."""
    try:
        if not text or not dest_lang:
            return ""
        # translator from googletrans earlier
        res = translator.translate(text, dest=dest_lang)
        # minimal backoff if receiving empty
        if not getattr(res, "text", None):
            time.sleep(0.2)
            return res.text or ""
        return res.text
    except Exception as e:
        # you can log e somewhere; for now return message so UI shows something
        return f"[Translation failed: {str(e)}]"


LANGUAGES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de"
}


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
    "bart": "sshleifer/distilbart-cnn-12-6",
    "t5": "google/flan-t5-small",
    "pegasus": "google/pegasus-xsum"
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
    return scores

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

def store_readability_db(user_email, text_content, scores, source="manual", filename=None):
        if filename is None:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        payload = {
            "user_email": user_email,
            "text_content": text_content,
            "scores": scores,
            "source": source,
            "filename": filename
        }
        try:
            resp = requests.post(f"{API_URL}/store_readability", json=payload, timeout=10)
            if resp.status_code == 200:
                st.success("Readability analysis stored successfully in DB")
            else:
                st.error(f"Failed to store readability: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to backend: {e}")



def show_dashboard():
    st.title("📊TextMorph's Dashboard")
    user_email = st.session_state.get("user_email")
    if not user_email:
        st.error("You must be logged in to use the dashboard")
        return

    
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs(["Readability Analysis", "Summarize", "Paraphrase","History","Dataset & Model Training", "Model Evaluation & Comparison"])

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
                    scores = analyze_and_show(manual_text)
                    store_readability_db(user_email, manual_text, scores, source="manual")

        
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
                            scores = analyze_and_show(content)
                            store_readability_db(user_email, content, scores, source="file", filename=uploaded_file.name)
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
                options=list(SUMMARIZATION_MODELS.keys()),  # ["bart", "t5", "pegasus"]
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
                            "bart": "sshleifer/distilbart-cnn-12-6",
                            "t5": "google/flan-t5-small",
                            "pegasus": "google/pegasus-xsum"
                        }

                        summarizer = load_summarizer(SUMMARIZATION_MODELS[model_choice])

                        wc_orig = len(original_text_display.split())
                        # ✅ Dynamic length scaling based on original word count
                        if summary_length == "short":
                            min_len, max_len = max(5, int(wc_orig * 0.1)), max(10, int(wc_orig * 0.3))
                        elif summary_length == "medium":
                            min_len, max_len = max(10, int(wc_orig * 0.2)), max(20, int(wc_orig * 0.5))
                        else:  # long
                            min_len, max_len = max(20, int(wc_orig * 0.3)), max(40, int(wc_orig * 0.8))


                        result = summarizer(original_text_display, min_length=min_len, max_length=max_len, do_sample=False)
                        summary_text_display = result[0]['summary_text'].strip()

                        wc_sum = len(summary_text_display.split())
                        compression = max(0, (1 - (wc_sum / wc_orig)) * 100) if wc_orig > 0 else 0


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
                            try:
                                rouge = evaluate.load("rouge")
                                scores = rouge.compute(
                                    predictions=[summary_text_display],
                                    references=[reference_input.strip()]
                                )

                                # ✅ move visualization inside the if block
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

                            except Exception as e:
                                st.error(f"ROUGE evaluation failed: {e}")
                        else:
                            st.info("⚠️ No reference summary provided, skipping ROUGE evaluation.")
                           
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
            "BART": "sshleifer/distilbart-cnn-12-6",
            "T5": "google/flan-t5-small",
            "PEGASUS": "google/pegasus-xsum"
        }
        selected_model = st.selectbox("Select Model", list(paraphrase_models.keys()))

        complexity_map = {"Beginner": 128, "Intermediate": 256, "Advanced": 512}
        max_len = complexity_map[complexity_level]

        complexity_prompt_map = {
            "Beginner": "Paraphrase the following text in simple and clear language suitable for beginners:",
            "Intermediate": "Paraphrase the following text with moderate complexity suitable for intermediate readers:",
            "Advanced": "Paraphrase the following text with advanced vocabulary and sentence structure suitable for expert readers:"
        }
        if "t5" in paraphrase_models[selected_model].lower():
            prompt_text = "paraphrase: " + original_text
        else:
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
        import pandas as pd
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
                resp = requests.get(f"{API_URL}/history/readability/{user_email}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        df = pd.DataFrame(data)

                        # Show overview table
                        st.dataframe(df[["id", "source", "filename", "created_at"]])

                        # Expanders for detailed view
                        for i, row in df.iterrows():
                            with st.expander(f"Entry {row['id']} ({row['created_at']})"):
                                st.markdown(f"**Source:** {row['source']}")
                                if row['filename']:
                                    st.caption(f"📂 File: {row['filename']}")

                                st.text_area(
                                    "Analyzed Text",
                                    value=row['text_content'][:1000],  # Preview first 1000 chars
                                    height=200,
                                    key=f"read_text_{i}"
                                )

                                st.markdown("**Readability Scores:**")
                                try:
                                    if isinstance(row['scores'], str):
                                        scores = json.loads(row['scores'])
                                    else:
                                        scores = row['scores']
                                    st.json(scores)
                                except Exception:
                                    st.warning("⚠️ Could not parse scores for this entry.")
                        df_expanded = df.copy()
                        try:
                            df_expanded["scores"] = df_expanded["scores"].apply(
                                lambda x: json.loads(x) if isinstance(x, str) else x
                            )
                            scores_df = pd.json_normalize(df_expanded["scores"])
                            df_expanded = pd.concat([df_expanded.drop(columns=["scores"]), scores_df], axis=1)
                        except Exception:
                            pass  # fallback if scores not parsable
                        csv_bytes = df_expanded.to_csv(index=False).encode()
                        st.download_button("Download CSV", data=csv_bytes, file_name="readability_history.csv")

                    else:
                        st.info("No readability analysis found.")
                else:
                    st.error(f"Failed to fetch readability history: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")

    with main_tab4:
        show_history(user_email)
        # ------------------ Dataset & Model Training ------------------
    # ------------------ Dataset & Model Training ------------------
    with main_tab5:
        st.markdown("### 📂 Dataset Management & Model Training")

        # ------------------ Step 1: Select Model ------------------
        model_type = st.radio("Select Model Type", ["Pre-existing Model", "Train From Scratch"])

        model_map = {
            "BART": "sshleifer/distilbart-cnn-12-6",
            "T5": "google/flan-t5-small",
            "PEGASUS": "google/pegasus-xsum"
        }

        model_name_or_path = None
        arch = None

        if model_type == "Pre-existing Model":
            selected_model = st.selectbox("Choose Pre-existing Model", list(model_map.keys()))
            model_name_or_path = model_map[selected_model]
        else:  # Train from scratch
            custom_model_name = st.text_input("Enter a name for your new model", value="my_new_model")
            arch = st.selectbox("Select Architecture Family", ["BART", "T5", "PEGASUS"])
            selected_model = f"{arch}-Scratch"

        # ------------------ Step 2: Select Task ------------------
        task_choice = st.radio("Select Task", ["Summarization", "Paraphrasing"])

        # ------------------ Step 3: Select Dataset ------------------
        summarization_datasets = ["cnn_dailymail"]
        paraphrasing_datasets = ["stsb", "paws", "mrpc"]

        if task_choice == "Summarization":
            dataset_choice = st.selectbox("Select Dataset", summarization_datasets)
        else:
            dataset_choice = st.selectbox("Select Dataset", paraphrasing_datasets)

        # ------------------ Step 4: Training Config ------------------
        epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
        batch_size = st.slider("Batch Size", min_value=1, max_value=64, value=16)
        max_samples = st.slider("Max samples to load", min_value=100, max_value=20000, value=2000, step=500)

        # ------------------ Training Helpers ------------------
        import threading, traceback, ijson, sys
        from contextlib import contextmanager
        from io import StringIO
        from transformers import TrainerCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments
        import evaluate



        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                np.where(labels != -100, labels, tokenizer.pad_token_id),
                skip_special_tokens=True
            )
            decoded_preds = [p.strip() for p in decoded_preds]
            decoded_labels = [l.strip() for l in decoded_labels]

            bleu_result = safe_bleu(decoded_preds, decoded_labels)
            bleu_score = bleu_result.get("bleu", 0.0)

            try:
                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels).get("rougeL", 0.0)
            except Exception:
                rouge_score = 0.0

            ppl = None
            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
                if losses:
                    ppl = math.exp(np.mean(losses))

            return {
                "bleu": bleu_score,
                "rougeL": rouge_score,
                "perplexity": ppl if ppl else float("nan")
            }

        @contextmanager
        def st_capture(output_area):
            old_write = sys.stdout.write
            buffer = StringIO()

            def new_write(b):
                old_write(b)
                buffer.write(b)
                output_area.text(buffer.getvalue())

            sys.stdout.write = new_write
            try:
                yield
            finally:
                sys.stdout.write = old_write

        class StreamlitCallback(TrainerCallback):
            def __init__(self, progress_bar, status_text, metrics_area):
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.metrics_area = metrics_area

            def on_train_begin(self, args, state, control, **kwargs):
                self.progress_bar.progress(0)
                self.status_text.text("🚀 Training started...")

            def on_step_end(self, args, state, control, **kwargs):
                if state.max_steps > 0:
                    progress = int((state.global_step / state.max_steps) * 100)
                    self.progress_bar.progress(progress)
                    self.status_text.text(
                        f"Step {state.global_step}/{state.max_steps} ({progress}%) - Epoch {state.epoch:.2f}"
                    )

            def on_train_end(self, args, state, control, **kwargs):
                self.progress_bar.progress(100)
                self.status_text.text("✅ Training finished!")

        def train_model(trainer, save_dir, log_area, progress_bar, status_text, metrics_area):
            try:
                with st_capture(log_area):
                    trainer.add_callback(StreamlitCallback(progress_bar, status_text, metrics_area))
                    trainer.train()

                    # ✅ Run evaluation manually after training
                    eval_results = trainer.evaluate()
                    bleu_score = eval_results.get("eval_bleu", None)
                    rouge_score = eval_results.get("eval_rougeL", None)
                    perplexity = eval_results.get("eval_perplexity", None)

                    st.subheader("📊 Final Evaluation Metrics")
                    st.write({
                        "BLEU": bleu_score,
                        "ROUGE-L": rouge_score,
                        "Perplexity": perplexity
                    })

                st.success("✅ Training complete!")
                st.info(f"Model saved to {save_dir}")

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.text("".join(traceback.format_exception(None, e, e.__traceback__)))

        # ------------------ Step 6: Train Button ------------------
        if st.button("🚀 Start Training"):
            try:
                with st.spinner("Preparing training..."):

                    # ------------------ Load model + tokenizer ------------------
                    if model_type == "Pre-existing Model":
                        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
                    else:
                        if arch == "BART":
                            from transformers import BartConfig, BartForConditionalGeneration
                            config = BartConfig()
                            model = BartForConditionalGeneration(config)
                            tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
                        elif arch == "T5":
                            from transformers import T5Config, T5ForConditionalGeneration
                            config = T5Config()
                            model = T5ForConditionalGeneration(config)
                            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                        elif arch == "PEGASUS":
                            from transformers import PegasusConfig, PegasusForConditionalGeneration
                            config = PegasusConfig()
                            model = PegasusForConditionalGeneration(config)
                            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

                    # ------------------ Load dataset from JSON ------------------
                    dataset_path = r"C:\Users\user\Desktop\pragna's codes\textsummarizationinfosys\datasets\cleaned_unified_dataset.json"
                    task_key = "summarization" if task_choice == "Summarization" else "paraphrase"

                    dataset = []
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        for entry in ijson.items(f, "item"):
                            if entry.get("task") == task_key and entry.get("dataset") == dataset_choice:
                                dataset.append(entry)
                            if len(dataset) >= max_samples:
                                break

                    if not dataset:
                        st.error(f"No entries found for {task_choice} - {dataset_choice}")
                        return

                    st.success(f"✅ Loaded {len(dataset)} entries for {task_choice} - {dataset_choice}")

                    hf_dataset = datasets.Dataset.from_list(dataset)


                    def preprocess(example):
                        inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)
                        targets = tokenizer(example["target"], truncation=True, padding="max_length", max_length=128)
                        inputs["labels"] = targets["input_ids"]
                        return inputs

                    with st.spinner("Tokenizing dataset..."):
                        tokenized_dataset = hf_dataset.map(preprocess, batched=True)

                    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
                    train_dataset = split["train"]
                    eval_dataset = split["test"]

                    # ------------------ TrainingArguments ------------------
                    save_dir = f"./results/{selected_model}_{task_choice}_{dataset_choice}"
                    training_args = Seq2SeqTrainingArguments(
                        output_dir=save_dir,
                        num_train_epochs=epochs,
                        per_device_train_batch_size=batch_size,
                        save_steps=5000,
                        save_total_limit=2,
                        logging_steps=50,
                        predict_with_generate=True,
                        report_to="none",
                        dataloader_pin_memory=False
                    )

                    trainer = Seq2SeqTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics
                    )

                    # ------------------ UI Elements ------------------
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    log_area = st.empty()
                    metrics_area = st.empty()
                    log_area.text("🚀 Training logs will appear here...\n")

                    # ------------------ Run in background ------------------
                    threading.Thread(
                        target=train_model,
                        args=(trainer, save_dir, log_area, progress_bar, status_text, metrics_area),
                        daemon=True
                    ).start()
                    st.info("🚀 Training started in the background... watch logs and progress ⬇️")

            except Exception as e:
                st.error(f"Training setup failed: {e}")
                st.text("".join(traceback.format_exception(None, e, e.__traceback__)))

    with main_tab6:
        st.markdown("### 📊 Model Evaluation & Comparison")
        def compute_perplexity(model, tokenizer, text):
            if not text.strip():
                return float("nan")
            encodings = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
            return math.exp(loss.item()) if loss is not None else float("nan")

        eval_task = st.radio("Select Task to Evaluate", ["Summarization", "Paraphrasing"], key="tab6_eval_task")

        # ---------------- Summarization Evaluation ----------------
        if eval_task == "Summarization":
            results_dir = "./results"
            available_models = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]

            if not available_models:
                st.warning("⚠️ No models found in results/. Train a model first.")
            else:
                selected_model = st.selectbox("Select Summarization Model", available_models, key="tab6_sum_model_select")

                uploaded_file = st.file_uploader(
                    "Upload evaluation dataset (CSV/TXT). Must have 'input' and 'target' columns if CSV.",
                    type=["csv", "txt"], key="tab6_sum_file_uploader"
                )

                eval_data = []
                if uploaded_file:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                        if "input" in df.columns and "target" in df.columns:
                            eval_data = df.to_dict(orient="records")
                        else:
                            st.error("CSV must have columns: 'input' and 'target'")
                    else:
                        lines = uploaded_file.getvalue().decode("utf-8").splitlines()
                        for line in lines:
                            if "|||" in line:
                                inp, tgt = line.split("|||")
                                eval_data.append({"input": inp.strip(), "target": tgt.strip()})

                # --- Run Summarization Evaluation ---
                if eval_data and st.button("🚀 Run Summarization Evaluation", key="tab6_run_sum_eval"):
                    st.success(f"✅ Loaded {len(eval_data)} evaluation samples")

                    model_path = os.path.join(results_dir, selected_model)
                    if "eval_tokenizer" not in st.session_state:
                        st.session_state["eval_tokenizer"] = AutoTokenizer.from_pretrained(model_path)

                    if "eval_model" not in st.session_state:
                        st.session_state["eval_model"] = AutoModelForSeq2SeqLM.from_pretrained(model_path)

                    tokenizer = st.session_state["eval_tokenizer"]
                    model = st.session_state["eval_model"]
                                        
                    rouge = evaluate.load("rouge")
                    
                    
                    sim_model = st.session_state.get("sim_model")
                    if sim_model is None:
                        sim_model = SentenceTransformer("all-MiniLM-L6-v2")
                        st.session_state["sim_model"] = sim_model


                    

                    task_prefix = "summarize: "
                    inputs = tokenizer([task_prefix + ex["input"] for ex in eval_data],
                                    return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer([ex["target"] for ex in eval_data],
                                    return_tensors="pt", padding=True, truncation=True)["input_ids"]

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=128,
                            num_beams=4,
                            early_stopping=True,
                            num_return_sequences=1,
                            length_penalty=1.0
                        )

                    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                    # --- Save results ---
                    st.session_state["sum_eval_results"] = {
                        "inputs": [ex["input"] for ex in eval_data],
                        "references": decoded_labels,
                        "generated": decoded_preds,
                    }
                    st.session_state["sum_eval_ready"] = True
                    st.success("✅ Summarization evaluation complete!")

                # 📌 Always display results if cached
                if st.session_state.get("sum_eval_ready", False):
                    inputs = st.session_state["sum_eval_results"]["inputs"]
                    refs = st.session_state["sum_eval_results"]["references"]
                    gens = st.session_state["sum_eval_results"]["generated"]

                    # --- Sample Comparisons ---
                    st.markdown("### 🔎 Sample Comparisons")
                    for i, (inp, ref, gen) in enumerate(zip(inputs, refs, gens)):
                        with st.expander(f"Sample {i+1}"):
                            col1, col2, col3 = st.columns(3)

                            # --- Original ---
                            with col1:
                                st.markdown("#### Original Text")
                                st.info(inp)
                                target_lang_inp = st.selectbox("Translate Original →", list(LANGUAGES.keys()), key=f"inp_lang_{i}")
                                key_inp = f"inp_trans_{i}"
                                if st.button("Translate Original", key=f"btn_inp_{i}"):
                                    st.session_state[key_inp] = safe_translate(inp, LANGUAGES[target_lang_inp])
                                if key_inp in st.session_state:
                                    st.text_area("Translated Original", st.session_state[key_inp], height=120, key=f"{key_inp}_area")

                            # --- Reference ---
                            with col2:
                                st.markdown("#### Reference Summary")
                                st.success(ref)
                                target_lang_ref = st.selectbox("Translate Reference →", list(LANGUAGES.keys()), key=f"ref_lang_{i}")
                                key_ref = f"ref_trans_{i}"
                                if st.button("Translate Reference", key=f"btn_ref_{i}"):
                                    st.session_state[key_ref] = safe_translate(ref, LANGUAGES[target_lang_ref])
                                if key_ref in st.session_state:
                                    st.text_area("Translated Reference", st.session_state[key_ref], height=120, key=f"{key_ref}_area")

                            # --- Generated ---
                            with col3:
                                st.markdown("#### Generated Summary")
                                st.warning(gen)
                                target_lang_gen = st.selectbox("Translate Generated →", list(LANGUAGES.keys()), key=f"gen_lang_{i}")
                                key_gen = f"gen_trans_{i}"
                                if st.button("Translate Generated", key=f"btn_gen_{i}"):
                                    st.session_state[key_gen] = safe_translate(gen, LANGUAGES[target_lang_gen])
                                if key_gen in st.session_state:
                                    st.text_area("Translated Generated", st.session_state[key_gen], height=120, key=f"{key_gen}_area")

                    # --- Metrics (averages across dataset) ---
                    st.markdown("### 📊 Average Metrics")
                    rows_out = []
                    for i, (inp, ref, gen) in enumerate(zip(inputs, refs, gens), start=1):
                        try:
                            model = st.session_state.get("eval_model")
                            tokenizer = st.session_state.get("eval_tokenizer")

                            # --- BLEU (safe) ---
                            bleu_scores = safe_bleu([gen], [ref])
                            # --- ROUGE ---
                            try:
                                rouge_scores = rouge.compute(predictions=[gen], references=[ref])
                            except Exception:
                                rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

                            # --- Perplexity ---
                            ppl_candidate = compute_perplexity(model, tokenizer, gen)
                            ppl_reference = compute_perplexity(model, tokenizer, ref)

                            # --- Semantic Similarity ---
                            from sentence_transformers import util 
                            sim_model = st.session_state.get("sim_model")
                            if sim_model is None:
                                sim_model = SentenceTransformer("all-MiniLM-L6-v2")
                                st.session_state["sim_model"] = sim_model

                            emb1 = sim_model.encode([gen], convert_to_tensor=True)
                            emb2 = sim_model.encode([ref], convert_to_tensor=True)
                            sim_score = float(util.cos_sim(emb1, emb2).item())

                            # --- Readability delta ---
                            try:
                                delta_flesch = textstat.flesch_reading_ease(gen) - textstat.flesch_reading_ease(inp)
                            except Exception:
                                delta_flesch = 0.0

                            # --- Append row ---
                            rows_out.append({
                                "input": inp,
                                "reference": ref,
                                "generated": gen,
                                "bleu": bleu_scores.get("bleu", 0.0),
                                "rouge1": rouge_scores.get("rouge1", 0.0),
                                "rouge2": rouge_scores.get("rouge2", 0.0),
                                "rougeL": rouge_scores.get("rougeL", 0.0),
                                "semantic_sim": sim_score,
                                "ppl_gen": ppl_candidate,
                                "delta_readability": delta_flesch,
                                "ppl_ref": ppl_reference,
                            })
                        except Exception as e:
                            st.warning(f"⚠️ Failed to process row {i}: {e}")
                            continue

                    rdf = pd.DataFrame(rows_out)
                    avg_metrics = {
                        "BLEU": rdf["bleu"].mean(),
                        "ROUGE-1": rdf["rouge1"].mean(),
                        "ROUGE-2": rdf["rouge2"].mean(),
                        "ROUGE-L": rdf["rougeL"].mean(),
                        "Semantic Sim": rdf["semantic_sim"].mean(),
                        "PPL (Gen)": rdf["ppl_gen"].mean(),
                        "PPL (Ref)": rdf["ppl_ref"].mean(),
                        "Δ Readability": rdf["delta_readability"].mean(),
                    }


                    colA, colB, colC = st.columns(3)
                    with colA: st.metric("BLEU", f"{avg_metrics['BLEU']:.2f}")
                    with colB: st.metric("ROUGE-L", f"{avg_metrics['ROUGE-L']:.2f}")
                    with colC: st.metric("Similarity", f"{avg_metrics['Semantic Sim']:.2f}")

                    colD, colE, colF = st.columns(3)
                    with colD: st.metric("Δ Readability", f"{avg_metrics['Δ Readability']:.2f}")
                    with colE: st.metric("PPL (Gen)", f"{avg_metrics['PPL (Gen)']:.2f}")
                    with colF: st.metric("PPL (Ref)", f"{avg_metrics['PPL (Ref)']:.2f}")

                    # 📊 Radar chart
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=[avg_metrics["BLEU"], avg_metrics["ROUGE-1"], avg_metrics["ROUGE-2"],
                        avg_metrics["ROUGE-L"], avg_metrics["Semantic Sim"]],
                        theta=["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Similarity"],
                        fill='toself'
                    ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # 📑 Detailed Results + Download
                    st.markdown("### 📑 Detailed Results")
                    st.dataframe(rdf)
                    st.download_button(
                        "📥 Download Results CSV",
                        data=rdf.to_csv(index=False).encode("utf-8"),
                        file_name="summarization_eval_results.csv",
                        mime="text/csv"
                    )

        # ---------------- Paraphrasing Evaluation ----------------
        elif eval_task == "Paraphrasing":
            st.markdown("### 📝 Paraphrase Dataset Evaluation")

            # Local fine-tuned model folders
            T5_PARAPHRASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","results", "saved_paraphrasing_model"))

            model_choice_eval = st.selectbox("Model", ["T5(finetuned)"], index=0, key="tab6_para_model_choice")
            MODEL_DIR_EVAL = T5_PARAPHRASE_DIR
            if not os.path.isdir(MODEL_DIR_EVAL):
                st.error(f"Model folder not found at {MODEL_DIR_EVAL}")
                st.stop()

            # Cache model
            @st.cache_resource
            def load_finetuned_model(model_dir):
                tok = AutoTokenizer.from_pretrained(model_dir)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
                return tok, mdl
            tokenizer, model = load_finetuned_model(MODEL_DIR_EVAL)

            src_para = st.radio("Data source", ["Upload CSV", "Upload TXT"], horizontal=True, key="tab6_para_source")

            selected_df = None
            if src_para == "Upload CSV":
                ds_file = st.file_uploader("Upload dataset CSV (columns: input_text,target_text or text,reference)", type=["csv"], key="tab6_para_file")
                if ds_file is not None:
                    try:
                        selected_df = pd.read_csv(ds_file)
                    except Exception as e:
                        st.error(f"Failed to read CSV: {e}")

            elif src_para == "Upload TXT":
                ds_file = st.file_uploader("Upload dataset TXT (format: input ||| reference)",
                                        type=["txt"], key="tab6_para_txt")
                if ds_file is not None:
                    try:
                        lines = ds_file.getvalue().decode("utf-8").splitlines()
                        rows = []
                        for line in lines:
                            if "|||" in line:
                                inp, ref = line.split("|||")
                                rows.append({"text": inp.strip(), "reference": ref.strip()})
                        selected_df = pd.DataFrame(rows)
                    except Exception as e:
                        st.error(f"Failed to read TXT: {e}")

            if selected_df is not None and not selected_df.empty:
                df = selected_df
                lower_cols = {c.lower(): c for c in df.columns}
                text_candidates = ["text", "input_text", "source", "article"]
                ref_candidates = ["reference", "target_text", "target", "paraphrase", "highlights"]
                text_col = next((lower_cols[c] for c in text_candidates if c in lower_cols), None)
                ref_col = next((lower_cols[c] for c in ref_candidates if c in lower_cols), None)
                if not text_col or not ref_col:
                    st.error("CSV must contain text and reference columns")
                    st.stop()
                df = df[[text_col, ref_col]].rename(columns={text_col: "text", ref_col: "reference"})

                total_rows = len(df)
                c1, c2 = st.columns(2)
                with c1:
                    start_row = st.number_input("Start row", min_value=1, max_value=max(1, total_rows), value=1, step=1)
                with c2:
                    default_end = min(total_rows, int(start_row) + 19)
                    end_row = st.number_input("End row", min_value=int(start_row), max_value=total_rows, value=default_end, step=1)

                s_idx = max(0, int(start_row) - 1)
                e_idx = min(total_rows, int(end_row))
                df = df.iloc[s_idx:e_idx].copy()

                if st.button("🚀 Run Paraphrase Evaluation", type="primary"):
                    rows_out = []
                    prog = st.progress(0)
                    
                    rouge = evaluate.load("rouge")

                    for i, (idx, row) in enumerate(df.iterrows(), start=1):
                        original = str(row["text"]) if pd.notna(row["text"]) else ""
                        reference = str(row["reference"]) if pd.notna(row["reference"]) else ""
                        if not original or not reference:
                            prog.progress(min(i / len(df), 1.0))
                            continue

                        inputs = tokenizer(original, return_tensors="pt", truncation=True, padding=True, max_length=256)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=128,
                                num_beams=5,
                                do_sample=False
                            )
                        gen = tokenizer.decode(outputs[0], skip_special_tokens=True)

                        bleu_scores = safe_bleu([gen], [reference])
                        try:
                            rouge_scores = rouge.compute(predictions=[gen], references=[reference])
                        except Exception:
                            rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

                        ppl_candidate = compute_perplexity(model, tokenizer, gen)
                        ppl_reference = compute_perplexity(model, tokenizer, reference)

                        from sentence_transformers import SentenceTransformer, util
                        sim_model = SentenceTransformer("all-MiniLM-L6-v2")

                        emb1 = sim_model.encode([gen], convert_to_tensor=True)
                        emb2 = sim_model.encode([reference], convert_to_tensor=True)
                        sim_score = float(util.cos_sim(emb1, emb2).item())
                        delta_flesch = textstat.flesch_reading_ease(gen) - textstat.flesch_reading_ease(reference)

                        rows_out.append({
                            "index": idx,
                            "text": original,
                            "generated": gen,
                            "reference": reference,
                            "bleu_1": bleu_scores.get("precisions", [0.0])[0],
                            "bleu_4": bleu_scores.get("precisions", [0.0, 0.0, 0.0, 0.0])[3],
                            "bleu": bleu_scores.get("bleu", 0.0),
                            "rouge1_f1": rouge_scores.get("rouge1", 0.0),
                            "rouge2_f1": rouge_scores.get("rouge2", 0.0),
                            "rougeL_f1": rouge_scores.get("rougeL", 0.0),
                            "semantic_sim": sim_score,
                            "ppl_candidate": ppl_candidate,    
                            "ppl_reference": ppl_reference,
                            "delta_flesch_reading_ease": delta_flesch
                        })
                        prog.progress(min(i / len(df), 1.0))

                    if rows_out:
                        rdf = pd.DataFrame(rows_out)

                        # ✅ Save results in session_state
                        st.session_state["para_eval_results"] = rdf
                        st.session_state["para_eval_ready"] = True
                        st.success("✅ Paraphrase evaluation complete")

                # --- Show results if cached ---
                if st.session_state.get("para_eval_ready", False):
                    rdf = st.session_state["para_eval_results"]

                    # --- Show a sample side-by-side comparison ---
                    sample = rdf.iloc[0]  # first row
                    col1, col2, col3 = st.columns(3)

                    # -------- Original Text --------
                    with col1:
                        st.markdown("#### Original Text")
                        st.info(sample["text"])

                        target_lang_inp = st.selectbox("Translate Original →", list(LANGUAGES.keys()), key="para_inp_lang")
                        key_para_inp = "para_inp_trans"
                        if st.button("Translate Original", key="btn_para_inp"):
                            with st.spinner("Translating original..."):
                                st.session_state[key_para_inp] = safe_translate(sample["text"], LANGUAGES[target_lang_inp])
                        if key_para_inp in st.session_state:
                            st.text_area("Translated Original", st.session_state[key_para_inp], height=120, key=f"{key_para_inp}_area")

                    # -------- Reference --------
                    with col2:
                        st.markdown("#### Reference")
                        st.success(sample["reference"])

                        target_lang_ref = st.selectbox("Translate Reference →", list(LANGUAGES.keys()), key="para_ref_lang")
                        key_para_ref = "para_ref_trans"
                        if st.button("Translate Reference", key="btn_para_ref"):
                            with st.spinner("Translating reference..."):
                                st.session_state[key_para_ref] = safe_translate(sample["reference"], LANGUAGES[target_lang_ref])
                        if key_para_ref in st.session_state:
                            st.text_area("Translated Reference", st.session_state[key_para_ref], height=120, key=f"{key_para_ref}_area")

                    # -------- Generated --------
                    with col3:
                        st.markdown("#### Generated")
                        st.warning(sample["generated"])

                        target_lang_gen = st.selectbox("Translate Generated →", list(LANGUAGES.keys()), key="para_gen_lang")
                        key_para_gen = "para_gen_trans"
                        if st.button("Translate Generated", key="btn_para_gen"):
                            with st.spinner("Translating generated..."):
                                st.session_state[key_para_gen] = safe_translate(sample["generated"], LANGUAGES[target_lang_gen])
                        if key_para_gen in st.session_state:
                            st.text_area("Translated Generated", st.session_state[key_para_gen], height=120, key=f"{key_para_gen}_area")

                    # --- Show full table & download ---
                    st.markdown("### 📑 Detailed Results")
                    st.dataframe(rdf)
                    st.download_button(
                        "📥 Download Results CSV",
                        data=rdf.to_csv(index=False).encode("utf-8"),
                        file_name="paraphrase_eval_results.csv",
                        mime="text/csv"
                    )
