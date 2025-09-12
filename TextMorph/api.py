from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
import mysql.connector
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import json
from datetime import datetime


dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

load_dotenv(dotenv_path)

app = FastAPI()

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB")
        )
        return conn
    except mysql.connector.Error as err:
        raise Exception(f"Database connection failed: {err}")

def save_file_to_db(user_email, filename, filetype, filesize, data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO uploaded_files (user_email, filename, filetype, filesize, filedata) VALUES (%s,%s,%s,%s,%s)",
            (user_email, filename, filetype, filesize, data)
        )
        conn.commit()
    except mysql.connector.Error as err:
        conn.rollback()
        raise err
    finally:
        cursor.close()
        conn.close()

@app.post("/upload")
async def upload_file(user_email: str = Form(...), uploaded_file: UploadFile = File(...)):
    try:
        data = await uploaded_file.read()
        save_file_to_db(user_email, uploaded_file.filename, uploaded_file.content_type, len(data), data)
        return {"message": "File uploaded successfully"}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"MySQL Error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

@app.get("/download/{file_id}")
def download_file(file_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT filename, filedata FROM uploaded_files WHERE id=%s", (file_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            filename, data = result
            return StreamingResponse(
                io.BytesIO(data),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        raise HTTPException(status_code=404, detail="File not found")

    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"MySQL Error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")
    
class SummaryEvaluation(BaseModel):
    user_email: str
    original_text: str
    summary_text: str
    model_used: str
    summary_length: str
    reference_summary: str = ""
    rouge_scores: dict = {}

@app.post("/store_evaluation")
def store_evaluation(evaluation: SummaryEvaluation):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO summaries 
            (user_email, original_text, summary_text, model_used, summary_length, reference_summary, rouge_scores, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            evaluation.user_email,
            evaluation.original_text,
            evaluation.summary_text,
            evaluation.model_used,
            evaluation.summary_length,
            evaluation.reference_summary,
            json.dumps(evaluation.rouge_scores), 
            datetime.now()
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "Evaluation stored successfully"}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"MySQL Error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")


class ParaphraseEvaluation(BaseModel):
    user_email: str
    original_text: str
    paraphrased_options: list   
    model_used: str
    creativity: float
    complexity_level: str   
    rouge_scores: list
    readability_scores: list


@app.post("/store_paraphrase")
def store_paraphrase(evaluation: ParaphraseEvaluation):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO paraphrases 
            (user_email, original_text, paraphrased_options, model_used, creativity, complexity_level, rouge_scores, readability_scores, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            evaluation.user_email,
            evaluation.original_text,
            json.dumps(evaluation.paraphrased_options),  
            evaluation.model_used,
            evaluation.creativity,
            evaluation.complexity_level,   # changed here
            json.dumps(evaluation.rouge_scores),         
            json.dumps(evaluation.readability_scores),   
            datetime.now()
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "Paraphrase stored successfully"}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"MySQL Error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

@app.get("/history/summaries/{user_email}")
def get_summary_history(user_email: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, original_text, summary_text, model_used, summary_length, reference_summary, rouge_scores, created_at
            FROM summaries
            WHERE user_email=%s
            ORDER BY created_at DESC
        """, (user_email,))
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching summary history: {str(e)}")


@app.get("/history/paraphrases/{user_email}")
def get_paraphrase_history(user_email: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, original_text, paraphrased_options, model_used, creativity, complexity_level, rouge_scores, readability_scores, created_at
            FROM paraphrases
            WHERE user_email=%s
            ORDER BY created_at DESC
        """, (user_email,))
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching paraphrase history: {str(e)}")


@app.get("/history/uploaded_files/{user_email}")
def get_readability_history(user_email: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, filename, filetype, filesize, uploaded_at
            FROM uploaded_files
            WHERE user_email=%s
            ORDER BY uploaded_at DESC
        """, (user_email,))
        files = cursor.fetchall()
        cursor.close()
        conn.close()
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching readability history: {str(e)}")
    
@app.get("/history/uploaded_files/content/{file_id}")
def get_uploaded_file_content(file_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT filename, filetype, filedata
            FROM uploaded_files
            WHERE id=%s
        """, (file_id,))
        file = cursor.fetchone()
        cursor.close()
        conn.close()

        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        content = ""
        if "text" in file["filetype"]:
            content = file["filedata"].decode("utf-8")
        elif "pdf" in file["filetype"]:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file["filedata"])) as pdf:
                content = "\n".join(page.extract_text() or "" for page in pdf.pages)
        else:
            content = f"Cannot display this file type: {file['filetype']}"

        return {"filename": file["filename"], "content": content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching file content: {str(e)}")

