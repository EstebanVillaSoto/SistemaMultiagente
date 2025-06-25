
"""


Sistema multiagente local con RAG completo (recuperación + generación) y FAISS.
Generación usando un modelo open-source de Hugging Face (sin API de pago).

Agentes:
  1. DataLoaderAgent: carga y preprocesa documentos (.txt, .pdf).
  2. IndexAgent: construye índice vectorial con FAISS.
  3. RetrieverAgent: realiza recuperación semántica local con filtrado por keywords.
  4. GenerationAgent: genera respuestas con modelo instruct de HuggingFace local (Flan-T5).

Tecnologías:
  - LangChain para estructura y chunking.
  - sentence-transformers para embeddings.
  - FAISS para base vectorial.
  - transformers para generación local instruct.

Este script **no requiere API key**.
Instalar:
  pip install langchain sentence-transformers faiss-cpu pdfplumber transformers
Ejecutar:
  python prueba1.py
"""

import os
from pathlib import Path
import pdfplumber
from typing import List, Dict
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


class DataLoaderAgent:
    """Carga y normaliza documentos .txt y .pdf"""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> List[Document]:
        docs: List[Document] = []
        for file in sorted(self.data_dir.iterdir()):
            if file.suffix.lower() == '.txt':
                text = file.read_text(encoding='utf8')
                docs.append(Document(page_content=text, metadata={'source': file.name}))
            elif file.suffix.lower() == '.pdf':
                with pdfplumber.open(file) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ''
                        docs.append(Document(
                            page_content=text,
                            metadata={'source': f'{file.name}-p{i+1}'},
                        ))
        return docs


class IndexAgent:
    """Construye el índice FAISS a partir de documentos"""
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def build(self, docs: List[Document]) -> FAISS:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        return FAISS.from_documents(chunks, self.embeddings)


class RetrieverAgent:
    """Recupera fragmentos relevantes con filtrado por keywords"""
    def __init__(self, index: FAISS, keywords: Dict[str, List[str]], k: int = 3):
        self.index = index
        self.k = k
        self.keywords = keywords

    def retrieve(self, query: str) -> List[Document]:
        q_lower = query.lower()
        for fname, keys in self.keywords.items():
            if any(k in q_lower for k in keys):
                hits = self.index.similarity_search(query, k=self.k)
                filtered = [d for d in hits if d.metadata.get('source','').startswith(fname)]
                if filtered:
                    return filtered
                break
        return self.index.similarity_search(query, k=self.k)


class GenerationAgent:
    # Genera respuesta final usando un modelo instruct de HuggingFace local
    def __init__(self, model: str = 'google/flan-t5-small'):
        self.generator = pipeline('text2text-generation', model=model, tokenizer=model)

    def generate(self, context: str, question: str) -> str:
        prompt = (
            f"Responde de forma clara y paso a paso a la pregunta usando este contexto:\n"
            f"Contexto:\n{context}\nPregunta: {question}\nRespuesta:"
        )
        outputs = self.generator(prompt, max_new_tokens=150, do_sample=False)
        text = outputs[0]['generated_text']
        return text.replace(prompt, '').strip()


def main():
    data_dir = 'docs'
    loader = DataLoaderAgent(data_dir)
    docs = loader.load()
    print(f"[DataLoaderAgent] {len(docs)} documentos cargados.")

    indexer = IndexAgent()
    index = indexer.build(docs)
    print("[IndexAgent] Índice vectorial listo.")

    keywords = {
        'network_troubleshooting.txt': ['conexión','red','ping','router'],
        'linux_commands.txt': ['linux','grep','comando','apt','df','ls','error','log','disco','uso','tamaño'],
        'office365_faq.txt': ['outlook','buzón','correo','firma'],
        'faq.txt': ['contraseña','windows','chrome']
    }
    retriever = RetrieverAgent(index, keywords)
    gen_agent = GenerationAgent(model='google/flan-t5-small')

    print("[Sistema] Listo. Escribe 'salir' para terminar.")
    while True:
        q = input("\nPregunta> ").strip()
        if q.lower() in ('salir','exit','quit'):
            break

        frags = retriever.retrieve(q)
        q_lower = q.lower()

        if any(k in q_lower for k in keywords['faq.txt']):
            frag = frags[0] if frags else None
            if frag:
                # Mostrar fragmento directamente sin generar
                print(
                    f"\nRespuesta:\n"
                    f"[{frag.metadata['source']}]\n"
                    f"{frag.page_content.strip()}"
                )
                continue

        context = '\n---\n'.join([d.page_content for d in frags])
        ans = gen_agent.generate(context, q)
        print(f"\nRespuesta generada:\n{ans}")

if __name__ == '__main__':
    main()

