LINK AL VIDEO:

LINK AL GITHUB:

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
