# test_rag.py in Backend/
from rag_service import RAGService

rag = RAGService()
answer = rag.process_and_ask("Bitcoin.pdf", "What is this paper about?")
print(answer)