from passlib.hash import bcrypt
from db import get_db_connection

# Admin credentials
name = "Main Admin"
email = "admin@textmorph.com"
password = "Admin@123"

# Hash the password
password_hash = bcrypt.hash(password)

# Insert into DB
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute(
    "INSERT INTO admins (name, email, password_hash) VALUES (%s, %s, %s)",
    (name, email, password_hash)
)
conn.commit()
cursor.close()
conn.close()

print("✅ Admin user created successfully!")
