from fastapi import FastAPI
import uvicorn
import psycopg2
from pydantic import BaseModel
import string
import random

conn=psycopg2.connect("dbname=db1 user=postgres password=123 port=5432 host=localhost")
cur=conn.cursor()

def create_tables():
    cur.execute('''
    CREATE TABLE info(
    id int PRIMARY KEY,
    name varchar(20)
                )
    ''')

    cur.execute('''
    CREATE TABLE History(
    id int REFERENCES info (id),
    question varchar(200),
    answer varchar(200)
                )
    ''')

    conn.commit()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# Models
class info(BaseModel):
    id:int
    name:str

class question(BaseModel):
    id:int
    question:str

app=FastAPI()

@app.get("/")
def welcome():
    return {"Message":"Welcome"}

@app.get("/gethistory")
def get_history(id:int):
    cur.execute('''
    SELECT * from History WHERE id=%s
''',(id,))
    
    if cur.fetchone():
        cur.execute('''
        SELECT * from History WHERE id=%s
        ''',(id,))
        records=cur.fetchall()
        return records
    else:
        return {"Message":"No Data available"}

@app.post("/question")
def take_ques(ques:question):
    cur.execute('''
    SELECT * from info WHERE id=%s
    ''',(ques.id,))
    if cur.fetchone(): 
        answer=id_generator(50,ques.question)
        cur.execute('''
        INSERT INTO History
        (id,question,answer)
        VALUES(%s,%s,%s)
        ''',(ques.id,ques.question,answer))
        conn.commit()
        return {"Answer":answer}
    else:
        return {"Message":"No id available"}

@app.post("/putinfo")
def put_info(Info:info):
    cur.execute('''
    SELECT * from info WHERE id=%s
    ''',(Info.id,))
    print()
    if not cur.fetchone():
        cur.execute('''
        INSERT INTO info
        (id,name)
        VALUES(%s,%s)
    ''',(Info.id,Info.name))
        conn.commit()
        return {"Message":"Succesfully Added"}
    else:
        return {"Message":"Id already available"}

@app.delete("/deleteinfo")
def delete_info(id:int):
    cur.execute('''
    SELECT * from info WHERE id=%s
    ''',(id,))

    if cur.fetchone():
        cur.execute('''
        DELETE FROM History
        WHERE id=%s
    ''',(id,))
        
        cur.execute('''
        DELETE FROM info
        WHERE id=%s
    ''',(id,))
        
        conn.commit()
        
        return {"Message":"Successfully Deleted"}
    else:
        return {"Message":"No ID exist"}
    
if __name__=='__main__':
    #create_tables()
    uvicorn.run(app=app,host='127.0.0.1',port=8000)