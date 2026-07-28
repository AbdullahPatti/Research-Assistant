from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGService:
    def __init__(self):
        self.emb = OllamaEmbeddings(model = "nomic-embed-text")
        self.llm = ChatOllama(model = "qwen2.5:7b", temperature = 0)
        self.vs = None

    def load_doc(self, path):
        return PyMuPDFLoader(path).load()

    def chunk(self, doc, size=1000, overlap=200):
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(doc)

    def ingest(self, pdf_path):
        doc = self.load_doc(pdf_path)
        chunks = self.chunk(doc)
        self.vs = Chroma.from_documents(chunks, self.emb, persist_directory="./chroma_data")
        return {"chunks_created": len(chunks)}

    def load_existing_store(self):
        self.vs = Chroma(persist_directory="./chroma_data", embedding_function=self.emb)

    def ask(self, question):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using only the context provided. Be concise.\n\nContext:\n{context}"),
            ("human", "{question}"),
        ])
        chain = (
            {"context": self.vs.as_retriever() | (lambda d: "\n\n".join(x.page_content for x in d)),
             "question": RunnablePassthrough()}
            | prompt | self.llm | StrOutputParser()
        )
        return chain.invoke(question)