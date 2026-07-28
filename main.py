from fastapi import FastAPI
from pydantic import BaseModel
from app.services.rag_service import RAGService

app = FastAPI(title="Research Assistant API")
rag = RAGService()

class IngestRequest(BaseModel):
    pdf_path: str

class AskRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest(req: IngestRequest):
    return rag.ingest(req.pdf_path)

@app.post("/ask")
def ask(req: AskRequest):
    answer = rag.ask(req.question)
    return {"answer": answer}