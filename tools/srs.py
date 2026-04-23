"""
Spaced Repetition System (SM-2 algorithm) for interview question scheduling.

Questions are stored in MongoDB with performance metadata. The SM-2 algorithm
computes the next review date based on quality of recall (0–5 scale).

SM-2 reference: https://www.supermemo.com/en/archives1990-2015/english/ol/sm2
"""
from __future__ import annotations

import logging
import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.tools import tool
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger("agent_interview_prep.srs")

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME = os.getenv("MONGO_DB_NAME", "agent_interview_prep")
_client: AsyncIOMotorClient | None = None


def _get_collection():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(_MONGO_URI)
    return _client[_DB_NAME]["srs_questions"]


def _sm2_next(easiness: float, interval: int, repetitions: int, quality: int) -> tuple[float, int, int]:
    """Apply SM-2 update step. Returns (new_easiness, new_interval, new_repetitions)."""
    if quality < 3:
        # Incorrect — reset repetitions, keep easiness
        return max(1.3, easiness), 1, 0

    new_easiness = max(1.3, easiness + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_repetitions = repetitions + 1
    if repetitions == 0:
        new_interval = 1
    elif repetitions == 1:
        new_interval = 6
    else:
        new_interval = round(interval * easiness)
    return new_easiness, new_interval, new_repetitions


@tool
async def record_attempt(
    user_id: str,
    question: str,
    topic: str,
    quality: int,
    session_id: str = "default",
) -> dict[str, Any]:
    """Record how well the user answered an interview question using SM-2 scheduling.
    quality: 0=blackout, 1=wrong, 2=wrong but close, 3=correct with difficulty, 4=correct, 5=perfect.
    Returns next review date and updated difficulty factor.
    """
    if not 0 <= quality <= 5:
        raise ValueError(f"quality must be 0–5, got {quality}")

    col = _get_collection()
    now = datetime.now(timezone.utc)

    stable_hash = int(hashlib.sha256(question.encode()).hexdigest(), 16) % (2**31)
    existing = await col.find_one({"user_id": user_id, "question_hash": stable_hash})

    easiness = existing.get("easiness_factor", 2.5) if existing else 2.5
    interval = existing.get("interval_days", 1) if existing else 1
    repetitions = existing.get("repetitions", 0) if existing else 0

    new_ef, new_interval, new_reps = _sm2_next(easiness, interval, repetitions, quality)
    next_review = now + timedelta(days=new_interval)

    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "question_hash": stable_hash,
        "question_preview": question[:200],
        "topic": topic,
        "easiness_factor": new_ef,
        "interval_days": new_interval,
        "repetitions": new_reps,
        "next_review_at": next_review,
        "last_quality": quality,
        "last_attempt_at": now,
        "total_attempts": (existing.get("total_attempts", 0) if existing else 0) + 1,
    }

    await col.replace_one(
        {"user_id": user_id, "question_hash": doc["question_hash"]},
        doc,
        upsert=True,
    )

    quality_labels = {0: "blackout", 1: "wrong", 2: "wrong (close)", 3: "correct (hard)", 4: "correct", 5: "perfect"}
    label = quality_labels.get(quality, str(quality))
    msg = (
        f"Recorded: {label}. Next review in **{new_interval} day{'s' if new_interval != 1 else ''}** "
        f"(EF={new_ef:.2f})."
    )
    if quality < 3:
        msg += " This question will repeat until you get it right."

    logger.info("SRS record_attempt: user=%s topic=%s quality=%d next_review=%s",
                user_id, topic, quality, next_review.date())
    return {"next_review_in_days": new_interval, "easiness_factor": new_ef, "message": msg}


@tool
async def get_due_questions(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return interview questions due for review today based on SM-2 spaced repetition scheduling.
    Returns a list of question previews with their topics and days overdue.
    """
    col = _get_collection()
    now = datetime.now(timezone.utc)
    cursor = col.find(
        {"user_id": user_id, "next_review_at": {"$lte": now}},
        sort=[("next_review_at", 1)],
        limit=limit,
    )
    results = []
    async for doc in cursor:
        days_overdue = max(0, (now - doc["next_review_at"]).days)
        results.append({
            "question_preview": doc.get("question_preview", ""),
            "topic": doc.get("topic", ""),
            "days_overdue": days_overdue,
            "last_quality": doc.get("last_quality", 0),
            "total_attempts": doc.get("total_attempts", 1),
        })
    return results


@tool
async def get_srs_stats(user_id: str) -> dict[str, Any]:
    """Return spaced repetition statistics for the user: total questions tracked, due today, mastered, and learning."""
    col = _get_collection()
    now = datetime.now(timezone.utc)

    total = await col.count_documents({"user_id": user_id})
    due = await col.count_documents({"user_id": user_id, "next_review_at": {"$lte": now}})
    mastered = await col.count_documents({"user_id": user_id, "easiness_factor": {"$gte": 2.8}, "repetitions": {"$gte": 3}})

    return {
        "total_questions": total,
        "due_today": due,
        "mastered": mastered,
        "learning": total - mastered,
    }
