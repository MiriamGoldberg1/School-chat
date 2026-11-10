# -*- coding: utf-8 -*-
"""יצירת מודלי Gemini (וקטורים ותשובות) לפי ההגדרות ב-config.py."""
import truststore

# אימות תעודות SSL דרך מאגר התעודות של Windows במקום של Python.
# נדרש ברשתות עם סינון (למשל נטפרי) שמשתמשות בתעודה משלהן.
truststore.inject_into_ssl()

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import config


def _check_api_key():
    if not config.GOOGLE_API_KEY:
        raise RuntimeError(
            "לא נמצא GOOGLE_API_KEY. יש ליצור קובץ .env בתיקיית הפרויקט "
            "עם השורה: GOOGLE_API_KEY=המפתח-שלך"
        )


def get_embed_model():
    """מודל ליצירת וקטורים (סעיף 7)."""
    _check_api_key()
    return GoogleGenAIEmbedding(
        model_name=config.EMBEDDING_MODEL,
        api_key=config.GOOGLE_API_KEY,
    )


def get_llm(model_name: str = config.LLM_MODEL, temperature: float = config.LLM_TEMPERATURE):
    """מודל ליצירת התשובה (סעיף 7)."""
    _check_api_key()
    return GoogleGenAI(
        model=model_name,
        api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
        # ניסיון חוזר אחד בלבד (2 ניסיונות): אם המודל חסום, עדיף לעבור מהר למודל הגיבוי
        max_retries=2,
    )
