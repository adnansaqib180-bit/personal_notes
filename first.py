import psycopg2

from psycopg2.extras import RealDictCursor

DB_SETTINGS = {
    'host' : 'localhost',
    'database' : 'fipkart_db',
    'user' : 'postgres',
    'password' : 123456789,
    'port' : 5432
}
print('conecting......')
connection = psycopg2.connect(**DB_SETTINGS)
cursor = connection.cursor(cursor_factory = RealDictCursor)
print('connected !!!')

query = 'SELECT * FROM flipcart LIMIT 3'
cursor.execute(query)

rows = cursor.fetchall()
a = rows[0]
print(type(a))