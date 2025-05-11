from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import os
import asyncio
import json
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

urls = ["https://coinmarketcap.com/"]

# Async function to handle document loading
async def fetch_data():
    loader = AsyncHtmlLoader(urls)
    docs = await loader.aload()
    return docs

# Run the async function
docs = asyncio.run(fetch_data())

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

content = docs_transformed[0].page_content


# Define extraction schema
schema = {
    "properties": {
        "product_name":{"type":"string"},
        "product_sale":{"type":"integer"},
        "product_price":{"type":"integer"},
    },
    "required": ["product_name", "product_sale","product_price"]
}

# Create extraction chain
chain = create_extraction_chain(schema=schema, llm=llm)

# Extract data
data = chain.invoke({"input": content})  # Ensure correct format
data=data["text"]
# Serialize JSON properly
json_object = json.dumps(data)

# Write to a JSON file
with open("data.json", "w") as outfile:
    outfile.write(json_object)