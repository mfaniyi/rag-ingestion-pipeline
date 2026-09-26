from fastapi import FastAPI, File, HTTPException, UploadFile
from rag_ingestion_pipeline.ingestion import validate_upload

app = FastAPI(title="RAG Ingestion Pipeline")

@app.get("/")
def root():
    return {"message": "RAG ingestion pipeline is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()

    try:
        validate_upload(file, len(content))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "filename": file.filename,
        "size": len(content),
        "message": "File passed validation",
    }
