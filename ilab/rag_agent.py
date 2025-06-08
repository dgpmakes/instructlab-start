import sys
import os
import faiss
import ollama
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Cargar texto desde PDF
def load_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Dividir texto en fragmentos
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Generar embeddings de los fragmentos
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings, model

# Crear índice FAISS
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Buscar fragmentos relevantes según la pregunta
def get_relevant_chunks(query, chunks, index, embed_model, top_k=3):
    q_emb = embed_model.encode([query])
    D, I = index.search(q_emb, top_k)
    return "\n".join([chunks[i] for i in I[0]])

# Enviar pregunta + contexto al modelo llama3
def generate_response_with_context(query, context):
    full_prompt = f"""Responde la siguiente pregunta usando el contexto proporcionado.

### Contexto:
{context}

### Pregunta:
{query}

### Respuesta:"""

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[
            {'role': 'user', 'content': full_prompt}
        ]
    )
    return response['message']['content']

# MAIN
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Debes indicar la ruta del documento. Ejemplo:")
        print("   python rag_agent.py documento.pdf")
        sys.exit(1)

    doc_path = sys.argv[1]

    if not os.path.exists(doc_path):
        print(f"❌ El archivo '{doc_path}' no existe.")
        sys.exit(1)

    if doc_path.lower().endswith('.pdf'):
        print("📄 Leyendo PDF...")
        raw_text = load_text_from_pdf(doc_path)
    else:
        print("📄 Leyendo archivo de texto...")
        with open(doc_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

    chunks = chunk_text(raw_text)

    print("🔎 Generando embeddings...")
    embeddings, embed_model = embed_chunks(chunks)

    print("📦 Indexando...")
    index = build_faiss_index(embeddings)

    query = input("❓ Pregunta al agente: ")
    print("📚 Buscando contexto relevante...")
    context = get_relevant_chunks(query, chunks, index, embed_model)

    print("🤖 Generando respuesta...")
    answer = generate_response_with_context(query, context)

    print("\n🧠 Respuesta del agente:\n")
    print(answer)

