# -*- coding: utf-8 -*-
"""
ממשק צ'אט לתלמידות לשאלות על תקנון בית הספר.

הרצה:  .venv\\Scripts\\streamlit.exe run app.py
"""
import streamlit as st

import config
from rag import SchoolRulesRAG

st.set_page_config(page_title="תקנון בית הספר", page_icon="📘")

# כיוון מימין לשמאל לכל הדף
st.markdown(
    """
    <style>
    .stApp, .stChatMessage, .stMarkdown, .stChatInput textarea, .stExpander {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag():
    """נטען פעם אחת בלבד ומשותף לכל המשתמשות."""
    return SchoolRulesRAG()


def friendly_error(error: Exception) -> str:
    text = str(error)
    if "429" in text and "PerDay" in text:
        return "הגענו למגבלת השאלות היומית של המכסה החינמית. נא לנסות שוב מחר."
    if "429" in text:
        return "הגענו למגבלת השאלות של המכסה החינמית. נא לנסות שוב בעוד דקה."
    if "503" in text:
        return "השרת של Gemini עמוס כרגע. נא לנסות שוב בעוד כמה שניות."
    return f"אירעה שגיאה: {text}"


def show_sources(message):
    """המקורות מוצגים מתוך ה-DB, כך שהם מופיעים גם אם המודל לא ציין אותם."""
    sources = message.get("sources")
    if not sources:
        return
    names = list(dict.fromkeys(s["source"] for s in sources))
    st.caption("📄 מקורות שנבדקו: " + ", ".join(names))
    with st.expander("הקטעים מהתקנון"):
        if message.get("model"):
            st.markdown(f"**המודל שענה:** {message['model']}")
        if message.get("search_query"):
            st.markdown(f"**שאלה מורחבת לחיפוש:** {message['search_query']}")
        for s in sources:
            st.markdown(f"**{s['source']}** · ציון התאמה {s['score']:.2f}")
            st.text(s["text"])


st.title("📘 שאלות על תקנון בית הספר")
st.caption("שאלי כל שאלה על כללי בית הספר: תלבושת, מבחנים, ארוחות, זמנים ועוד.")

with st.sidebar:
    st.header("הגדרות המערכת")
    st.caption("לשינוי: config.py")
    st.markdown(
        f"""
- **חלוקה:** {config.CHUNKING_METHOD} ({config.CHUNK_SIZE})
- **חפיפה:** {config.CHUNK_OVERLAP if config.USE_CHUNK_OVERLAP else "ללא"}
- **Top K:** {config.TOP_K}
- **חיפוש:** {config.SEARCH_MODE}
- **הרחבת שאלה:** {"כן" if config.USE_QUERY_EXPANSION else "לא"}
- **מודל וקטורים:** {config.EMBEDDING_MODEL}
- **מודל תשובות:** {config.LLM_MODEL}
- **מודלי גיבוי:** {", ".join(config.LLM_FALLBACK_MODELS) or "ללא"}
"""
    )
    if st.button("שיחה חדשה"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("כתבי כאן את השאלה..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפשת בתקנון..."):
            try:
                result = load_rag().ask(question)
                found = result.found
                message = {
                    "role": "assistant",
                    "content": result.answer,
                    "model": result.model,
                    "search_query": result.search_query if result.search_query != question else None,
                    # כשאין תשובה בתקנון, אין טעם להציג קטעים לא רלוונטיים
                    "sources": [
                        {"source": n.node.metadata["source"], "score": n.score, "text": n.node.text}
                        for n in result.sources
                    ] if found else [],
                }
            except Exception as e:
                message = {"role": "assistant", "content": friendly_error(e)}
        st.markdown(message["content"])
    st.session_state.messages.append(message)
