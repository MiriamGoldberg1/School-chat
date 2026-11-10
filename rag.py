# -*- coding: utf-8 -*-
"""
שלב השאילתה: שאלה -> (הרחבה) -> חיפוש ב-ChromaDB -> תשובה מ-Gemini.

בדיקה משורת הפקודה:
    .venv\\Scripts\\python.exe rag.py "מה מותר ללבוש בחורף?"
"""
import re
import sys
from dataclasses import dataclass, field

import chromadb
from llama_index.core import PromptTemplate, VectorStoreIndex, get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from models import get_embed_model, get_llm

NOT_FOUND_ANSWER = config.NOT_FOUND_ANSWER

# ההוראה למודל נבנית מההגדרות ב-config.py (ASSISTANT_ROLE, ANSWER_RULES).
# {context_str} = קטעי התקנון שנשלפו, {query_str} = שאלת התלמידה


PREFIX_HE = config.CREATIVE_ANSWER_PREFIX["he"]
PREFIX_YI = config.CREATIVE_ANSWER_PREFIX["yi"]
PREFIX_OTHER = config.CREATIVE_ANSWER_PREFIX["other"]

# {creative_prefix} בכללים מוחלף בהוראה שמתאימה לכל שפה
_creative_prefix = (
    f'(if the answer is in Hebrew) exactly this sentence: "{PREFIX_HE}" - '
    f'or (if the answer is in Yiddish) exactly this sentence: "{PREFIX_YI}" - '
    f'or (in any other language) this sentence translated into that language: "{PREFIX_OTHER}"'
)
_rules = "\n".join(
    f"{i}. {rule.replace('{creative_prefix}', _creative_prefix)}"
    for i, rule in enumerate(config.ANSWER_RULES, 1)
)
QA_PROMPT = PromptTemplate(config.QA_TEMPLATE).partial_format(role=config.ASSISTANT_ROLE, rules=_rules)

LANGUAGE_FIX_PROMPT = PromptTemplate(config.LANGUAGE_FIX_INSTRUCTIONS)

_LANG_LINE = re.compile(rf"^\s*{re.escape(config.LANGUAGE_TAG)}\s*(.+?)\s*$", re.MULTILINE)


def _has_hebrew_letters(text: str) -> bool:
    return re.search(r"[֐-׿]", text) is not None


def _split_language_line(answer: str, question: str) -> tuple[str, str]:
    """מפריד את שורת LANG: מהתשובה. מחזיר (שפה, תשובה בלי השורה)."""
    match = _LANG_LINE.search(answer)
    if not match:
        # המודל לא כתב את השורה - הערכה גסה לפי האותיות של השאלה
        return ("Hebrew" if _has_hebrew_letters(question) else "the language of the question"), answer
    return match.group(1).strip(" .*"), (answer[:match.start()] + answer[match.end():]).strip()


# אותיות מאלפביתים שלא אמורים להופיע בתשובה: יוונית, קירילית, ערבית, שפות הודו.
# המודל משבץ אותן לפעמים בטעות באמצע מילה (למשל "αυθενטישע" ביידיש)
_STRAY_LETTERS = re.compile(r"[Ͱ-ϿЀ-ӿ؀-ۿऀ-෿]")
_STRAY_LETTER_LANGS = ("greek", "russian", "ukrainian", "arabic", "persian", "hindi", "urdu")


def _has_stray_letters(language: str, answer: str) -> bool:
    lang = language.lower()
    return not any(name in lang for name in _STRAY_LETTER_LANGS) and _STRAY_LETTERS.search(answer) is not None


def _needs_language_fix(language: str, answer: str) -> bool:
    """האם התשובה מעורבבת או משובשת, ולא כתובה כולה בשפת השאלה."""
    if _has_stray_letters(language, answer):
        return True
    lang = language.lower()
    if "hebrew" in lang or "עברית" in lang:
        return False
    # בשפה שאינה עברית לא אמור להופיע המשפט העברי (גם לא ביידיש)
    if PREFIX_HE in answer:
        return True
    # ביידיש (ובשפות נוספות באותיות עבריות) אי אפשר לזהות עברית לפי האותיות
    hebrew_script = any(name in lang for name in ("yiddish", "ladino", "יידיש", "אידיש"))
    if not hebrew_script and _has_hebrew_letters(answer):
        return True
    # המשפט האנגלי הקבוע לא תורגם לשפת השאלה
    return "english" not in lang and PREFIX_OTHER in answer

# סעיף 8: ניסוח מחדש של השאלה לפני יצירת הוקטור
EXPANSION_PROMPT = PromptTemplate(config.EXPANSION_INSTRUCTIONS)


@dataclass
class RagResult:
    answer: str
    search_query: str                    # הטקסט שלפיו בוצע החיפוש (מורחב או מקורי)
    sources: list[NodeWithScore] = field(default_factory=list)
    model: str = ""                      # המודל שענה בפועל (ראשי או גיבוי)
    found: bool = True                   # False = התשובה לא נמצאה בתקנון
    language: str = ""                   # שפת השאלה, כפי שהמודל זיהה אותה


def _is_quota_error(error: Exception) -> bool:
    return getattr(error, "code", None) == 429 or "429" in str(error)


def _is_overloaded_error(error: Exception) -> bool:
    return getattr(error, "code", None) == 503 or "503" in str(error)


class SchoolRulesRAG:
    def __init__(self):
        self.embed_model = get_embed_model()
        # המודל הראשי ואחריו מודלי הגיבוי, לפי הסדר ב-config.py
        self.models = [config.LLM_MODEL, *config.LLM_FALLBACK_MODELS]
        self._llms = {}
        self._synthesizers = {}
        # המודל שממנו מתחילים. מתקדם קדימה כשלמודל נגמרת המכסה
        self._first_model = 0

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            collection = client.get_collection(config.CHROMA_COLLECTION)
        except Exception:
            raise RuntimeError("ה-DB ריק. יש להריץ קודם: .venv\\Scripts\\python.exe ingest.py")

        # וקטורים ממודל אחד לא ניתנים להשוואה לוקטורים ממודל אחר
        db_model = (collection.metadata or {}).get("embedding_model")
        if db_model and db_model != config.EMBEDDING_MODEL:
            raise RuntimeError(
                f"ה-DB נבנה עם {db_model} אבל בהגדרות מוגדר {config.EMBEDDING_MODEL}. "
                "יש להריץ מחדש את ingest.py"
            )

        index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=collection),
            embed_model=self.embed_model,
        )
        self.retriever = index.as_retriever(**self._retriever_kwargs())

    def _llm(self, model, temperature=config.LLM_TEMPERATURE):
        if (model, temperature) not in self._llms:
            self._llms[(model, temperature)] = get_llm(model, temperature)
        return self._llms[(model, temperature)]

    def _synthesizer(self, model):
        if model not in self._synthesizers:
            self._synthesizers[model] = get_response_synthesizer(
                llm=self._llm(model), text_qa_template=QA_PROMPT, response_mode="compact"
            )
        return self._synthesizers[model]

    def _with_fallback(self, call):
        """
        מריץ call(model) על המודל הראשי, ובמקרה של מכסה (429) או עומס (503)
        עובר למודל הבא. מחזיר (תוצאה, שם המודל שהצליח).
        """
        last_error = None
        for i in range(self._first_model, len(self.models)):
            model = self.models[i]
            try:
                return call(model), model
            except Exception as e:
                if _is_quota_error(e):
                    # המכסה לא תתחדש בקרוב - מדלגים על המודל הזה גם בשאלות הבאות
                    self._first_model = i + 1
                elif not _is_overloaded_error(e):
                    raise
                last_error = e
        # כל המודלים נכשלו. מתחילים מחדש מהראשי בפעם הבאה (אולי המכסה התחדשה)
        self._first_model = 0
        raise last_error

    @staticmethod
    def _retriever_kwargs():
        """סעיפים 6 ו-9: כמות התוצאות וצורת החיפוש."""
        kwargs = {"similarity_top_k": config.TOP_K}
        if config.SEARCH_MODE == "mmr":
            kwargs["vector_store_query_mode"] = "mmr"
            kwargs["vector_store_kwargs"] = {"mmr_threshold": config.MMR_THRESHOLD}
        elif config.SEARCH_MODE != "similarity":
            raise ValueError(f"SEARCH_MODE לא מוכר: {config.SEARCH_MODE!r}")
        return kwargs

    def expand_question(self, question: str) -> str:
        prompt = EXPANSION_PROMPT.format(question=question)
        response, _ = self._with_fallback(lambda model: self._llm(model).complete(prompt))
        return response.text.strip() or question

    def ask(self, question: str) -> RagResult:
        search_query = self.expand_question(question) if config.USE_QUERY_EXPANSION else question
        nodes = self.retriever.retrieve(search_query)
        if not nodes:
            return RagResult(NOT_FOUND_ANSWER, search_query, found=False)

        # החיפוש לפי השאלה המורחבת, אבל התשובה ניתנת לשאלה המקורית
        response, model = self._with_fallback(
            lambda model: self._synthesizer(model).synthesize(question, nodes)
        )
        # המודל כותב קודם שורת "LANG: <שפה>". מפרידים אותה מהתשובה
        language, answer = _split_language_line(str(response).strip(), question)
        # הודעת "לא נמצא" מתורגמת לשפת השאלה, ולכן מזהים אותה לפי הסימון ולא לפי הטקסט
        found = config.NOT_FOUND_MARKER not in answer
        answer = answer.replace(config.NOT_FOUND_MARKER, "").strip()
        # עד שני ניסיונות תיקון: לפעמים אחרי הניסיון הראשון עדיין נשארות אותיות משובשות
        for _ in range(2):
            if not (config.USE_LANGUAGE_FIX and _needs_language_fix(language, answer)):
                break
            answer = self.fix_language(question, answer, language)
        return RagResult(answer, search_query, nodes, model, found, language)

    def fix_language(self, question: str, answer: str, language: str) -> str:
        prompt = LANGUAGE_FIX_PROMPT.format(question=question, answer=answer, language=language)
        # שכתוב לשפה אחרת הוא משימה מדויקת ולא יצירתית: טמפרטורה 0 מונעת תווים משובשים
        response, _ = self._with_fallback(lambda model: self._llm(model, temperature=0).complete(prompt))
        return response.text.strip() or answer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]) or "מה מותר ללבוש בחורף?"
    result = SchoolRulesRAG().ask(question)

    print(f"שאלה: {question}")
    if result.search_query != question:
        print(f"שאלה מורחבת: {result.search_query}")
    print(f"\nתשובה ({result.model}):\n{result.answer}\n")
    print(f"קטעים שנשלפו ({config.SEARCH_MODE}, top_k={config.TOP_K}):")
    for n in result.sources:
        print(f"  {n.score:.3f}  {n.node.metadata['source']}: {n.node.text[:70].replace(chr(10), ' | ')}")


if __name__ == "__main__":
    main()
