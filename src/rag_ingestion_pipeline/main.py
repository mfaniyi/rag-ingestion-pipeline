from fastapi import FastAPI

app = FastAPI(title="RAG Ingestion Pipeline")

@app.get("/")
def root():
    return {"message": "RAG ingestion pipeline is running"}


