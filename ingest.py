"""Build the retrieval index for Incident Copilot, and search it.

Pipeline:   documents -> chunks -> embeddings -> vector store -> search

No LLM is involved. This stage only finds relevant text. Week 2 adds reasoning
on top of what is found here.

Usage:
    python ingest.py                            # build the index
    python ingest.py --search "pods OOMKilled"  # query it
"""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).parent / "data"
STORE = Path(__file__).parent / "chroma_db"
COLLECTION = "incident_knowledge"

# 384-dimensional embeddings, ~80MB, runs locally on CPU. Small and fast enough
# that we can re-embed the whole corpus while experimenting.
MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_WORDS = 120
CHUNK_OVERLAP = 25

_model = None


def model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts):
    """Turn text into vectors. The same model must be used for documents and
    queries, otherwise the vectors live in different spaces and similarity is
    nonsense."""
    return model().encode(texts, normalize_embeddings=True).tolist()


def chunk(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word windows.

    Why chunk at all: an embedding is one fixed-size vector, so the longer the
    text, the more meaning gets averaged away. Smaller chunks give sharper
    matches but less context.

    Why overlap: a hard cut can split a symptom from its cause. Repeating a few
    words at the boundary means a sentence straddling the split still appears
    whole in one chunk.

    Most documents here are short and come back as a single chunk. The function
    exists so chunk size becomes a variable we can measure in week 3.
    """
    words = text.split()
    if len(words) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


def incident_text(inc):
    """Flatten an incident into the text we search over.

    The resolution is included on purpose: when triaging a new incident, the
    most useful past document is one that says what actually fixed it.
    """
    return (
        f"Incident {inc['id']} on {inc['service']} ({inc['severity']}). "
        f"{inc['title']}. Impact: {inc['impact']}. "
        f"Symptoms: {inc['symptoms']}. "
        f"Logs: {inc['log_excerpt']} "
        f"Resolution: {inc['resolution']}"
    )


def runbook_text(rb):
    return f"Runbook {rb['id']}: {rb['title']}. {rb['body']}"


def build_documents():
    """Load both sources and return parallel lists of ids, texts and metadata.

    Metadata is what makes citation possible: every chunk carries the id of the
    document it came from, so an answer can point back at its source.
    """
    incidents = json.loads((DATA / "incidents.json").read_text())
    runbooks = json.loads((DATA / "runbooks.json").read_text())

    ids, texts, metas = [], [], []

    for inc in incidents:
        for n, piece in enumerate(chunk(incident_text(inc))):
            ids.append(f"{inc['id']}#{n}")
            texts.append(piece)
            metas.append({
                "doc_id": inc["id"],
                "doc_type": "incident",
                "service": inc["service"],
                "severity": inc["severity"],
                "root_cause_category": inc["root_cause_category"],
                "runbook_id": inc["runbook_id"],
            })

    for rb in runbooks:
        for n, piece in enumerate(chunk(runbook_text(rb))):
            ids.append(f"{rb['id']}#{n}")
            texts.append(piece)
            metas.append({
                "doc_id": rb["id"],
                "doc_type": "runbook",
                "service": "",
                "severity": "",
                "root_cause_category": rb["root_cause_category"],
                "runbook_id": rb["id"],
            })

    return ids, texts, metas


def build_index():
    client = chromadb.PersistentClient(path=str(STORE))

    # Delete and recreate so a rebuild never leaves stale chunks behind.
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
        # State the distance metric explicitly. Chroma's default is squared L2, and
    # relying on that default means a library upgrade could silently change
    # every score. With cosine, distance = 1 - cosine similarity.
    collection = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    ids, texts, metas = build_documents()
    print(f"embedding {len(texts)} chunks with {MODEL_NAME} ...")
    collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embed(texts))

    print(f"indexed {collection.count()} chunks into {STORE.name}/")
    return collection


def get_collection():
    client = chromadb.PersistentClient(path=str(STORE))
    return client.get_collection(COLLECTION)


def search(query, k=5):
    """Return the k most similar chunks, with their source for citation."""
    result = get_collection().query(query_embeddings=embed([query]), n_results=k)

    hits = []
    for doc, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        # Chroma returns a distance (lower is closer). Convert to a similarity
        # score so bigger always means better, which is less error-prone later.
        hits.append({
            "doc_id": meta["doc_id"],
            "doc_type": meta["doc_type"],
            "root_cause_category": meta["root_cause_category"],
            "similarity": round(1 - distance, 3),
            "text": doc,
        })
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", metavar="QUERY")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.search:
        for n, hit in enumerate(search(args.search, args.k), 1):
            print(f"\n{n}. [{hit['doc_id']}] {hit['doc_type']} "
                  f"similarity={hit['similarity']} "
                  f"cause={hit['root_cause_category']}")
            print(f"   {hit['text'][:160]}...")
    else:
        build_index()


if __name__ == "__main__":
    main()
