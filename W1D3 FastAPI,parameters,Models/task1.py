# query params to body.
import uvicorn
from pydantic import BaseModel
import psycopg2
from fastapi import FastAPI

class info(BaseModel):
    id:int
    name:str
    adress:str
    city:str

class infoupdate(BaseModel):
    id:int
    city:str

class infodelete(BaseModel):
    id:int

conn = psycopg2.connect("dbname=db1 user=postgres password=123 host=localhost port=5433")
cur = conn.cursor()

app=FastAPI()

@app.get("/")
def read_root():
    return {"Message":"Welcome"}

@app.get("/getinfo")
def read_info():
    cur.execute("SELECT * from Persons")
    records=cur.fetchall()
    if records:
        return {"Message":records}
    else:
        return {"Message":"No data available"}
    

@app.post("/addinfo")
def read_item(Info:info): #here query parameters are send in API
    cur.execute('''
    SELECT * from Persons WHERE PersonID=%s
''',(Info.id,))
    
    if not cur.fetchone():
        cur.execute('''
        INSERT into Persons 
        (PersonID,Name,Address,City)
        VALUES (%s,%s,%s,%s)
    ''',(Info.id,Info.name,Info.adress,Info.city))
        conn.commit()
        return {"Message":"Record  succesfully added"}
    else:
        return {"Message":"Data already available"}

@app.put("/updateinfo")
def update_info(Info:infoupdate):
    cur.execute('''
    SELECT * from Persons WHERE PersonID=%s
''',(Info.id,))
    
    if cur.fetchone():
        cur.execute('''
        UPDATE Persons 
        SET City=%s
        WHERE PersonID=%s
    ''',(Info.city,Info.id))
        conn.commit()
        return {"Message":"Record  succesfully Updated"}
    else:
       return {"Message":"No such record available"}

@app.delete("/deleteinfo")
def delete_info(Info:infodelete):
    cur.execute('''
    SELECT * from Persons WHERE PersonID=%s
''',(Info.id,))
    

    if cur.fetchone():
        cur.execute('''
        DELETE FROM Persons 
        WHERE PersonID=%s
    ''',(Info.id,))
        conn.commit()
        return {"Message":"Record  succesfully deleted"}
    else:
        return {"Message":"No such record available"}


if __name__=='__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)