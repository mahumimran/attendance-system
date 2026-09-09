"""
Attendance business logic shared by both the AI-recognition route and the
manual-attendance route, so duplicate-prevention and status rules live in
exactly one place.
"""
from datetime import date as date_type, datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from backend.models.models import Attendance, Student


def get_existing_attendance(db: Session, student_pk: int, for_date: date_type) -> Optional[Attendance]:
    return (
        db.query(Attendance)
        .filter(Attendance.student_id_fk == student_pk, Attendance.date == for_date)
        .first()
    )


def mark_attendance(
    db: Session,
    student_pk: int,
    status: str,
    method: str,
    for_date: Optional[date_type] = None,
    confidence: Optional[float] = None,
):
    """
    Inserts an attendance record, enforcing "one record per student per day".

    Returns (record, created: bool, message: str).
    If a record already exists:
      - AI method: refuses silently with a friendly message (no duplicate).
      - Manual method: updates the existing record's status (teachers are
        explicitly allowed to correct manual attendance for the day).
    """
    today = for_date or datetime.now().date()
    now_time = datetime.now().time()

    existing = get_existing_attendance(db, student_pk, today)

    if existing:
        if method == "Manual":
            existing.status = status
            existing.time = now_time
            existing.method = method
            existing.confidence = confidence
            db.commit()
            db.refresh(existing)
            return existing, False, "Attendance updated."
        return existing, False, "Attendance already recorded for today."

    record = Attendance(
        student_id_fk=student_pk,
        date=today,
        time=now_time,
        status=status,
        method=method,
        confidence=confidence,
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        # Race condition backstop: DB-level UNIQUE constraint caught a
        # concurrent duplicate insert that slipped past the check above.
        db.rollback()
        existing = get_existing_attendance(db, student_pk, today)
        return existing, False, "Attendance already recorded for today."

    db.refresh(record)
    return record, True, "Attendance marked successfully."
