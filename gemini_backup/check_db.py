import sqlite3
db = sqlite3.connect('trading_data.db')
cursor = db.cursor()
cursor.execute('SELECT * FROM app_settings WHERE id=1')
res = cursor.fetchone()
print(res)
db.close()
