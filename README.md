# TextMorph

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-orange)](https://streamlit.io/)

TextMorph is a web-based application that allows users to upload text or input it manually for **readability analysis**,**text summarization** and **paraphrasing**. It integrates a **FastAPI backend**, **MySQL database**, and a **Streamlit frontend** to provide a smooth user experience for processing textual content.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Features

- Manual text input with readability analysis
- Upload `.txt` files for analysis
- Summarization of large texts using pre-trained models
- Paraphrasing of text for alternative phrasing and better clarity
- Database storage of user information and uploaded files
- Download uploaded files
- Easy-to-use web interface with Streamlit
- Multi-tab dashboard for manual and file input
- Color-coded readability score visualization

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/PavaniKoduri/TextMorph.git
cd TextMorph
```

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate the virtual environment**

```bash
source venv/bin/activate
```

4. **Install Dependencies**

```bash
pip install -r requirements.txt
```

# Environment Variables

TextMorph uses environment variables to configure the database connection securely. These variables should be stored in a `.env` file located in the root directory of the project.

## Create a `.env` File

In the root folder of your project, create a file named `.env` and add the following variables:

```env
# MySQL database host (usually localhost)
MYSQL_HOST=localhost

# MySQL username
MYSQL_USER=root

# MySQL password
MYSQL_PASSWORD=yourpassword

# Name of the MySQL database used by TextMorph
MYSQL_DB=textmorph_user
```

# Database Setup

TextMorph requires a MySQL database to store user information and uploaded files. Follow the steps below to set it up.

## 1. Create the Database

Connect to your MySQL server (e.g., using MySQL Workbench, phpMyAdmin, or terminal) and run:

```sql
CREATE DATABASE textmorph_user;
USE textmorph_user;
```

## Create Required Tables

After creating the database, you need to create the tables that TextMorph will use. Run the following SQL commands:

```sql
-- Users table to store user information
CREATE TABLE users (
    mailid VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age_group VARCHAR(50),
    lang_pre VARCHAR(50),
    password_hash VARCHAR(255) NOT NULL
);

-- Uploaded files table to store files uploaded by users
CREATE TABLE uploaded_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    filetype VARCHAR(100),
    filesize INT,
    filedata LONGBLOB NOT NULL,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_email) REFERENCES users(mailid)
);
```

## Running the Application

Once the environment and database are set up, you can start the backend and frontend as follows:

### 1. Start the FastAPI Backend

```bash
uvicorn TextMorph.api:app --reload
```

### 2. Start the Streamlit Frontend

```bash
streamlit run TextMorph/dashboard.py
```

## Usage

After running both the backend and frontend, you can interact with TextMorph through the Streamlit dashboard.

### 1. Manual Text Input

- Go to the **Manual Input** tab.
- Enter your text in the text area provided.
- Click **Save & Analyze Manual Text**.
- The text will be saved locally and analyzed for readability.
- Readability scores will be displayed in a color-coded bar chart:
  - **Green**: Easy to read
  - **Yellow**: Moderate
  - **Red**: Hard to read

### 2. File Upload

- Go to the **Upload File** tab.
- Upload a `.txt` file by clicking **Browse**.
- Click **Upload** to send the file to the FastAPI backend.
- Once uploaded, the text content will be analyzed for readability.
- You can also download previously uploaded files if needed.

### 3. Summarization (Future)

- Large texts can be summarized using pre-trained models integrated into TextMorph (if implemented).
- Simply input text manually or upload a file and click **Summarize** (button provided in dashboard when feature is available).

### 4. Paraphrasing (Future)

- Go to the **Paraphrasing** tab in the dashboard.
- Enter or upload text you want to rephrase.
- Click **Paraphrase** to generate alternative versions of the text.
- Useful for improving writing style, avoiding repetition, or simplifying text.

### Notes

- Ensure you are logged in (user email required) before using the dashboard.
- Uploaded files and manual inputs are stored in the MySQL database and local directory respectively.
- Always keep the backend running while using the frontend.

## Folder Structure

The project is organized as follows:

```plaintext
TextMorph/
│
├── TextMorph/ # Backend and core application code
│ ├── api.py # FastAPI endpoints for file upload/download
│ ├── db.py # Database connection utility
│ └── __init__.py
│ └── login.py
│ └── forgotpassword.py
│ └── signup.py
│
├── webapp/ # Streamlit frontend code
│ └── dashboard.py # Main dashboard for manual input and file upload
│ └── profile.py
│
├── manual_inputs/ # Folder to save manual text inputs
│
├── .env # Environment variables for database connection
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # License file
```

## Technologies Used

- **Python 3.10** – Programming language used for backend and frontend.
- **FastAPI** – Framework for building the backend API.
- **Streamlit** – Library to build the interactive frontend dashboard.
- **MySQL** – Relational database for storing user info and uploaded files.
- **textstat** – Python library for calculating readability scores.
- **Matplotlib** – Library for generating color-coded bar charts of readability scores.
- **Requests** – Library to make HTTP requests from Streamlit to FastAPI.
- **python-dotenv** – For loading environment variables from a `.env` file.
- **Transformers** – For paraphrasing and summarization models (Pegasus, BART, FLAN-T5).
- **NLTK** – For text preprocessing in paraphrasing and summarization.

## License

This project is under development and not yet licensed. A license will be added in the future.

```

```
