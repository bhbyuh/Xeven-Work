import pandas as pd
import jsonlines
import json

def function(file_path):
    excel_data = pd.ExcelFile(file_path)

    # Extract the names of all sheets in the Excel file
    sheet_names = excel_data.sheet_names

    # Load data from the second sheet (index 1) into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_names[1])

    # Remove rows where the 'Text' column has null values
    df = df[df['Text'].notna()]

    text=df["Text"].to_list()

    # For the first 10 records, parse the JSON strings in the 'Text' column
    # Extract the 'text' field from the first dictionary in each JSON
    for i in range(10):
        text[i] = json.loads(text[i])[0]['text']

    with jsonlines.open("Spam_Dataset.jsonl", mode='w') as jsonl_file:
        for i in range(len(text)):
            values = []
            values.append({"role": "system","content": "You are a spam analyst specializing in detecting spam messages. Your task is to classify the message as 'spam' or 'normal'."})
            values.append({"role": "user", "content": text[i]})
            values.append({"role": "assistant", "content": "spam"})
            json_data = {"messages":values}
            jsonl_file.write(json_data)
    
if __name__=="__main__":
    file_path = "Frauds - Compromised templates v2.xlsx"
    function(file_path)