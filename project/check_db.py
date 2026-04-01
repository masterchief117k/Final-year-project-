import sqlite3
conn = sqlite3.connect('database.db')
c = conn.cursor()
c.execute("SELECT name, emp_id, image_path FROM employees")
for row in c.fetchall():
    print(f"Name: '{row[0]}', EmpID: '{row[1]}', Path: '{row[2]}'")
conn.close()
