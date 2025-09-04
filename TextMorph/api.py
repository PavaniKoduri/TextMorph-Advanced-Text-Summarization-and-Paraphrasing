from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from .db import get_db_connection
import io
app = FastAPI()
def save_file_to_db(user_email, filename, filetype, filesize, data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO uploaded_files (user_email, filename, filetype, filesize, filedata) VALUES (%s,%s,%s,%s,%s)",
        (user_email, filename, filetype, filesize, data))
    conn.commit()
    cursor.close()
    conn.close()
@app.post("/upload")
async def upload_file(user_email: str = Form(...), uploaded_file: UploadFile = File(...)):
    data = await uploaded_file.read()
    save_file_to_db(user_email, uploaded_file.filename, uploaded_file.content_type, len(data), data)
    return {"message": "File uploaded successfully"}
@app.get("/download/{file_id}")
def download_file(file_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, filedata FROM uploaded_files WHERE id=%s", (file_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
        filename, data = result
        return StreamingResponse(io.BytesIO(data), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={filename}"})
    return {"error": "File not found"}