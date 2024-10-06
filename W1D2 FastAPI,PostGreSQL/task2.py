import psycopg2

# Connect to your postgres DB
conn = psycopg2.connect(dbname="db1", user="postgres",host="localhost",port=5433,password="123")

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a query
cur.execute('''CREATE TABLE Persons (
    PersonID int primary key,
    Name varchar(255),
    Address varchar(255),
    City varchar(255)
);''')

conn.commit()