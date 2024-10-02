import uvicorn
import psycopg2
from fastapi import FastAPI

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
def read_item(id:int,name:str,adress:str,city:str):
    cur.execute('''
    SELECT * from Persons WHERE PersonID=%s
''',(id,))
    
    if not cur.fetchone():
        cur.execute('''
        INSERT into Persons 
        (PersonID,Name,Address,City)
        VALUES (%s,%s,%s,%s)
    ''',(id,name,adress,city))
        conn.commit()
        return {"Message":"Record  succesfully added"}
    else:
        return {"Message":"Data already available"}

@app.put("/updateinfo")
def update_info(id:int,city:str):
    cur.execute('''
    SELECT * from Persons WHERE PersonID=%s
''',(id,))
    
    if cur.fetchone():
        cur.execute('''
        UPDATE Persons 
        SET City=%s
        WHERE PersonID=%s
    ''',(city,id))
        conn.commit()
        return {"Message":"Record  succesfully Updated"}
    else:
       return {"Message":"No such record available"}

@app.delete("/deleteinfo")
def delete_info(id:int):
    cur.execute('''
    SELECT * from Persons WHERE PersonID=%s
''',(id,))
    

    if cur.fetchone():
        cur.execute('''
        DELETE FROM Persons 
        WHERE PersonID=%s
    ''',(id,))
        conn.commit()
        return {"Message":"Record  succesfully deleted"}
    else:
        return {"Message":"No such record available"}


if __name__=='__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)