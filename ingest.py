# -*- coding: utf-8 -*-
"""
שלב הקליטה: קורא את קבצי ה-WORD, מחלק ל-chunks, יוצר וקטורים ושומר ב-ChromaDB.

הרצה:  .venv\\Scripts\\python.exe ingest.py
יש להריץ מחדש בכל שינוי בקבצי התקנון, בהגדרות החלוקה או במודל הוקטורים.
"""
import re
import sys
import unicodedata

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from models import get_embed_model

# תווי כיווניות נסתרים (RLM, LRM וכו') שמופיעים לעיתים בשמות קבצים בעברית
_BIDI_CHARS = re.compile(r"[‎‏‪-‮⁦-⁩]")
# שורות שאינן תוכן ואין טעם לחפש בהן
_NOISE_LINES = {'בס"ד', "בס''ד", "בעזה\"י"}


def clean_name(file_name: str) -> str:
    """שם קובץ נקי לתצוגה כמקור: בלי סיומת ובלי תווים נסתרים."""
    name = file_name.rsplit(".", 1)[0]
    return _BIDI_CHARS.sub("", unicodedata.normalize("NFC", name)).strip()


def clean_text(text: str) -> str:
    """הסרת שורות רעש ושורות ריקות. כל פסקה ב-WORD נשארת שורה נפרדת."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(l for l in lines if l and l not in _NOISE_LINES)


def load_documents():
    documents = SimpleDirectoryReader(
        input_dir=str(config.DATA_DIR),
        required_exts=config.SUPPORTED_EXTENSIONS,
    ).load_data()

    for doc in documents:
        doc.set_content(clean_text(doc.text))
        # שם הקובץ הוא נושא הכללים (למשל "תלבושת"). הוא נכלל גם בטקסט שממנו
        # נוצר הוקטור וגם בהקשר שנשלח ל-LLM, כדי שכל chunk "יידע" לאיזה נושא הוא שייך
        doc.metadata = {"source": clean_name(doc.metadata["file_name"])}
        doc.excluded_embed_metadata_keys = []
        doc.excluded_llm_metadata_keys = []
    return documents


def build_splitter(embed_model):
    """בחירת שיטת החלוקה לפי סעיפים 3-4 ב-config.py."""
    overlap = config.CHUNK_OVERLAP if config.USE_CHUNK_OVERLAP else 0

    if config.CHUNKING_METHOD == "tokens":
        return SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=overlap)

    if config.CHUNKING_METHOD == "words":
        # אותו מנגנון כמו טוקנים, אבל "טוקן" = מילה (פיצול לפי רווחים)
        return TokenTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=overlap,
            tokenizer=str.split,
        )

    if config.CHUNKING_METHOD == "semantic":
        return SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=config.SEMANTIC_BUFFER_SIZE,
            breakpoint_percentile_threshold=config.SEMANTIC_BREAKPOINT_PERCENTILE,
            # כל פסקה ב-WORD היא כלל נפרד, ולכן משמשת כ"משפט" בהשוואת המשמעות
            sentence_splitter=lambda t: [l + "\n" for l in t.splitlines() if l.strip()],
        )

    raise ValueError(f"CHUNKING_METHOD לא מוכר: {config.CHUNKING_METHOD!r}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    embed_model = get_embed_model()

    documents = load_documents()
    print(f"נטענו {len(documents)} קבצים מתוך {config.DATA_DIR}")

    nodes = build_splitter(embed_model).get_nodes_from_documents(documents)
    print(f"שיטת חלוקה: {config.CHUNKING_METHOD} | נוצרו {len(nodes)} chunks\n")
    for i, node in enumerate(nodes, 1):
        preview = node.text.replace("\n", " | ")
        print(f"[{i}] ({node.metadata['source']}, {len(node.text.split())} מילים) {preview[:120]}")

    # בנייה מחדש של ה-collection מאפס, כדי שלא יישארו chunks מהרצה קודמת
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass  # ה-collection עוד לא קיים (הרצה ראשונה)
    collection = client.create_collection(
        config.CHROMA_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            # נשמר כדי לוודא בזמן החיפוש שמשתמשים באותו מודל וקטורים
            "embedding_model": config.EMBEDDING_MODEL,
        },
    )

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=collection)
    )
    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    print(f"\nנשמרו {collection.count()} וקטורים ב-ChromaDB: {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
