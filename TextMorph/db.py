import mysql.connector
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
# Load environment variables from .env file
load_dotenv(dotenv_path)

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
        print(f"Database connection failed: {err}")
        return None