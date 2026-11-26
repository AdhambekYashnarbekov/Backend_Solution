from fastapi import FastAPI, HTTPException, Response

from app import llm  # noqa: F401  (you will need it)
from app.chunking import chunk_text  # noqa: F401  (you will need it)


def create_app() -> FastAPI:
    """Return a NEW app with an EMPTY in-memory store.

    The tests call create_app() once per test, so state must not leak
    between app instances (no module-level store).
    """
    app = FastAPI()
    store = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

# TODO: POST /ingest
    @app.post("/ingest")
    def ingest(body: dict):
        documents = body.get("documents")
        if not isinstance(documents, list) or len(documents) == 0:
            raise HTTPException(status_code=422, detail="Invalid documents")

        total_chunks = 0

        for doc in documents:
            if not isinstance(doc, dict):
                raise HTTPException(status_code=422, detail="Invalid document")

            doc_id = doc.get("id")
            text = doc.get("text")

            if not isinstance(doc_id, str) or doc_id.strip() == "":
                raise HTTPException(status_code=422, detail="Invalid id")
            if not isinstance(text, str) or text.strip() == "":
                raise HTTPException(status_code=422, detail="Invalid text")

            chunks = chunk_text(text)
            doc_chunks = []

            for i in range(len(chunks)):
                doc_chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": doc_id + ":" + str(i),
                    "text": chunks[i],
                    "vector": llm.embed(chunks[i]),
                })

            store[doc_id] = doc_chunks
            total_chunks += len(doc_chunks)

        return {"ingested": len(documents), "chunks": total_chunks}

    # TODO: POST /ask
    @app.post("/ask")
    def ask(body: dict):
        question = body.get("question")
        top_k = body.get("top_k", 3)

        if not isinstance(question, str) or question.strip() == "":
            raise HTTPException(status_code=422, detail="Invalid question")
        if type(top_k) is not int or top_k < 1 or top_k > 10:
            raise HTTPException(status_code=422, detail="Invalid top_k")
        if len(store) == 0:
            raise HTTPException(status_code=404, detail="No documents ingested")

        q_vec = llm.embed(question)
        scored = []

        for doc_id in store:
            for chunk in store[doc_id]:
                score = 0.0
                for i in range(len(q_vec)):
                    score += q_vec[i] * chunk["vector"][i]

                scored.append((score, chunk["doc_id"], chunk["chunk_id"], chunk["text"]))

        scored.sort(reverse=True)
        top_chunks = scored[:top_k]

        contexts = []
        sources = []
        for score, doc_id, chunk_id, text in top_chunks:
            contexts.append(text)
            sources.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": score,
                "text": text,
            })

        answer = llm.generate(question, contexts)
        return {"answer": answer, "sources": sources}

    # TODO: GET /documents
    @app.get("/documents")
    def get_documents():
        docs = []
        for doc_id in sorted(store.keys()):
            docs.append({"id": doc_id, "chunks": len(store[doc_id])})
        return {"documents": docs}

    # TODO: DELETE /documents/{doc_id}
    @app.delete("/documents/{doc_id}")
    def delete_documents(doc_id: str):
        if doc_id not in store:
            raise HTTPException(status_code=404, detail="Document not found")
        del store[doc_id]
        return Response(status_code=204)

    return app

    
    
    

app = create_app()
