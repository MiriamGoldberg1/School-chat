# -*- coding: utf-8 -*-
"""
קובץ ההגדרות של מערכת ה-RAG לשאלות על תקנון בית הספר.

כל נקודות ההחלטה מקובץ ההוראות מרוכזות כאן.
לשינוי התנהגות המערכת - משנים את הערכים בקובץ זה בלבד.

שימו לב: שינוי בהגדרות החלוקה (סעיפים 3-4) או במודל הוקטורים (סעיף 7)
מחייב בנייה מחדש של ה-DB (הרצה מחדש של ingest.py).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# סעיף 5 - API KEY
# המפתח נשמר בקובץ .env (לא בקוד!) בשורה: GOOGLE_API_KEY=...
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------------
# נתיבים
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"            # כאן שמים את קבצי ה-WORD של התקנון
SUPPORTED_EXTENSIONS = [".docx"]

# ---------------------------------------------------------------------------
# סעיף 1 - VECTOR DB: ChromaDB (נשמר מקומית על הדיסק)
# ---------------------------------------------------------------------------
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION = "school_rules"

# ---------------------------------------------------------------------------
# סעיף 3 - שיטת חלוקת הידע ל-chunks
#   "tokens"   - לפי כמות טוקנים (SentenceSplitter, לא חותך משפט באמצע)
#   "words"    - לפי כמות מילים
#   "semantic" - לפי משמעות (SemanticSplitter) - הכי איכותי והכי יקר,
#                כי מחשב embedding לכל משפט בזמן החלוקה
# ---------------------------------------------------------------------------
CHUNKING_METHOD = "tokens"

# גודל chunk: בטוקנים (tokens) או במילים (words). לא רלוונטי ל-semantic.
CHUNK_SIZE = 512

# הגדרות לשיטת semantic בלבד:
# אחוזון הסף לפיצול - גבוה יותר = פחות פיצולים = chunks גדולים יותר
SEMANTIC_BREAKPOINT_PERCENTILE = 95
# כמה משפטים סמוכים לקבץ יחד בעת השוואת המשמעות
SEMANTIC_BUFFER_SIZE = 1

# ---------------------------------------------------------------------------
# סעיף 4 - חפיפה בין chunks
#   True  - כל chunk מתחיל בסוף ה-chunk הקודם (שומר על הקשר)
#   False - ללא חפיפה
# גודל החפיפה באותה יחידה כמו CHUNK_SIZE. בשיטת semantic אין חפיפה.
# ---------------------------------------------------------------------------
USE_CHUNK_OVERLAP = True
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# סעיף 7 - מודלים (Gemini)
# ---------------------------------------------------------------------------
# מודל ליצירת וקטורים
EMBEDDING_MODEL = "gemini-embedding-001"
# מודל ליצירת התשובה
LLM_MODEL = "gemini-3.5-flash-lite"
# מודלי גיבוי: אם למודל הראשי נגמרה המכסה (429) או שהשרת עמוס (503),
# המערכת עוברת אוטומטית למודל הבא ברשימה. לכל מודל יש מכסה חינמית נפרדת.
# רשימה ריקה [] = ללא גיבוי
LLM_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
# טמפרטורה =רמת היצירתיות
LLM_TEMPERATURE = 0.8

# ---------------------------------------------------------------------------
# סעיף 6 - TOP K: כמה chunks לשלוף מה-DB בכל חיפוש
# ---------------------------------------------------------------------------
TOP_K = 5

# ---------------------------------------------------------------------------
# סעיף 8 - הרחבת השאלה (Query Expansion)
#   True  - לפני החיפוש, ה-LLM מנסח מחדש ומרחיב את השאלה (קריאת API נוספת)
#   False - מחפשים לפי השאלה כפי שנשאלה
# ---------------------------------------------------------------------------
USE_QUERY_EXPANSION =True

# ---------------------------------------------------------------------------
# סעיף 9 - צורת החיפוש ב-VECTOR DB
#   "similarity" - דמיון קוסינוס: מחזיר את ה-chunks הכי דומים לשאלה
#   "mmr"        - Maximal Marginal Relevance: דומים לשאלה אבל שונים זה מזה
#                  (מונע החזרה של כמה chunks כמעט זהים)
# ---------------------------------------------------------------------------
SEARCH_MODE = "similarity"
# ל-mmr בלבד: 1.0 = רק רלוונטיות, 0.0 = רק גיוון
MMR_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# הנחיות למודל (PROMPT) - כאן מגדירים את כללי התשובה
# אין צורך להריץ מחדש את ingest.py אחרי שינוי כאן.
# ---------------------------------------------------------------------------
# התשובה כשהמידע לא נמצא בתקנון
NOT_FOUND_ANSWER = "לא מצאתי מידע על כך בתקנון בית הספר. מומלץ לפנות למחנכת או למזכירות."
# סימון טכני שהמודל מוסיף כשאין תשובה. הקוד מזהה אותו (בכל שפה) ומסיר אותו לפני ההצגה
NOT_FOUND_MARKER = "[NOT_FOUND]"
# המשפט שפותח את התשובה היצירתית כשאין מידע בתקנון. מוכנס לכללים במקום {creative_prefix}:
# לתשובה בעברית / ביידיש - הגרסה המתאימה כלשונה, לכל שפה אחרת - תרגום של הגרסה האנגלית
# (ביידיש המודל מתרגם את המשפט האנגלי באופן משובש, ולכן יש לו גרסה קבועה)
CREATIVE_ANSWER_PREFIX = {
    "he": "אין לי מקור מידע מוסמך לענות על השאלה הזו, אבל לדעתי,",
    "yi": "איך האָב נישט קיין באַגלייבטן מקור פון אינפאָרמאַציע צו ענטפֿערן אויף דער פֿראַגע, אָבער לויט מײַן מיינונג,",
    "other": "I do not have an authoritative source of information to answer this question, but in my opinion,",
}

# תפקיד הבוט
ASSISTANT_ROLE = "You are an assistant who answers students' questions about the school regulations."

# כללי התשובה - אפשר להוסיף, למחוק או לשנות שורות
ANSWER_RULES = [
    "Detect the language of the current question (the 'Question:' line below) and write your whole reply only in that language "
    "(for example: English question = English answer). Ignore the language of the passages, of these instructions "
    "and of any earlier question. This rule also applies to the 'not found' message below.",
    # "Answer only based on the passages above. Do not use general knowledge or add details that are not written in them.",
    # כלל 2: שני מקרים נפרדים - יש תשובה בתקנון / אין תשובה בתקנון
    "First decide whether the passages contain the answer, then follow exactly one case:\n"
    "   (a) The passages contain the answer: answer only from the passages. "
    f"Do NOT write {NOT_FOUND_MARKER}, do NOT add a creative answer and do NOT add an opinion.\n"
    f"   (b) Only if the passages do not contain the answer: start your reply with {NOT_FOUND_MARKER}, "
    "then tell the student in a creative, warm and varied way (not a fixed sentence) that the school regulations "
    "do not contain information about this, then write {creative_prefix}, "
    "followed by a creative answer.",
    "Provide clear answers",
    "Never show the user the original source documents under any circumstances.",
    "If a question is grammatically meaningless or unclear, try to infer its intended meaning and respond: 'Did you mean ...?'",
]

# המבנה המלא של ההנחיה שנשלחת למודל. השדות בסוגריים מסולסלים מתמלאים אוטומטית:
#   {role}        = ASSISTANT_ROLE
#   {rules}       = ANSWER_RULES (ממוספרים)
#   {context_str} = קטעי התקנון שנשלפו מה-DB
#   {query_str}   = שאלת התלמידה
# השורה LANGUAGE_TAG: המודל כותב בה קודם את שם שפת השאלה (למשל "LANG: Yiddish").
# הקוד משתמש בה כדי לבדוק את שפת התשובה ומסיר אותה לפני ההצגה. אין למחוק אותה מההנחיה.
LANGUAGE_TAG = "LANG:"
QA_TEMPLATE = (
    "{role}\n"
    "Below are passages from the school regulations. Each passage starts with a 'source' line naming its chapter.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Rules:\n{rules}\n\n"
    "Question: {query_str}\n\n"
    "First write one line in exactly this format: " + LANGUAGE_TAG + " <the language of the question, "
    "for example Hebrew, Yiddish, English, French>. Note that Yiddish is also written in Hebrew letters - "
    "do not confuse it with Hebrew.\n"
    "Then, on the next lines, write the answer entirely in that language.\n"
)

# ההנחיה להרחבת השאלה (סעיף 8). {question} יוחלף בשאלת התלמידה
EXPANSION_INSTRUCTIONS = (
    "A student asked a question about the school regulations (on topics such as dress code, meals, exams, "
    "schedules, lateness, prohibited items, activities).\n"
    "Rephrase the question as a complete and clear question in Hebrew, and add synonyms and related terms "
    "that may appear in the regulations. Do not answer the question.\n"
    "Return only the expanded question, on a single line.\n\n"
    "Question: {question}\n"
    "Expanded question: "
)

# תיקון שפה: כשהתשובה יצאה מעורבבת (עברית באמצע, או משפט שלא תורגם),
# המערכת שולחת את התשובה למודל פעם נוספת לשכתוב בשפת השאלה (קריאת API נוספת, רק כשצריך).
# {question} = שאלת התלמידה, {answer} = התשובה המעורבבת, {language} = שפת השאלה
USE_LANGUAGE_FIX = True
LANGUAGE_FIX_INSTRUCTIONS = (
    "The answer below must be written entirely in {language}, "
    "but some parts of it are in another language, or contain broken words with letters from other alphabets "
    "(for example Greek or Arabic letters).\n"
    "Rewrite the whole answer in correct {language}, using only the alphabet of {language}. "
    "Keep the meaning, content and tone exactly. "
    "A Hebrew term from the regulations may be kept in parentheses right after its translation.\n"
    "Return only the rewritten answer.\n\n"
    "Question: {question}\n\n"
    "Answer: {answer}\n\n"
    "Rewritten answer: "
)
