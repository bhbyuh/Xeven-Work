import uvicorn 
from fastapi import FastAPI

app = FastAPI()

#sample {'12':{'name':'ALi','email':'@gmail.com'}}
info={'12':{'name':'ALi','email':'@gmail.com'}}

@app.get("/")
def read_root(id:int):
    return {"Message":"Welcome"}

@app.get("/getinfo")
def read_info(userid:int):
    if(userid in info):
        id=info[userid]
        if(id is not None):
            return {"Message":info[userid]}
    else:
        return {"Message":"No info gainst that id"}

@app.post("/addinfo")
def read_item(id:str,name:str,email:str):
    keys=info.keys()
    if (id in keys):
        return {"Message":"Info already Exist"}
    else:
        info[id]={"name":name,"email":email}
        print(info)

@app.put("/updateinfo")
def update_info(id:str,email:str):
    if(id in info):
        info[id]["email"]=email
        print(info)
        return {"Message":"Succesfuly Updated"}
    else:
        return {"Message":"No info gainst that id"}

@app.delete("/deleteinfo")
def delete_info(id:str):
    
    if(id in info):
        info.pop(id)
        print(info)
        return {"Message":"Succesfuly Deleted"}
    else:
        return {"Message":"No info gainst that id"}

if __name__=='__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)

