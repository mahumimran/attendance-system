"""
ORM models.

Design notes:
- `face_embedding` stores a JSON-encoded list of 128 floats (the
  face_recognition library's embedding size) rather than a raw image.
  This keeps biometric storage minimal and avoids re-computing embeddings
  on every recognition pass.
- `UniqueConstraint("student_id", "date")` on Attendance enforces
  "one attendance record per student per day" at the database level,
  as a backstop to the application-level check in services/attendance.py.
"""
from sqlalchemy import (
    Column, Integer, String, Float, Date, Time, DateTime,
    ForeignKey, UniqueConstraint, Text, Boolean
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(32), unique=True, index=True, nullable=False)
    name = Column(String(120), nullable=False)
    class_name = Column(String(50), nullable=False)
    section = Column(String(10), nullable=False)
    roll_number = Column(String(20), nullable=False)

    # JSON-encoded 128-d embedding vector; null until face registration completes.
    face_embedding = Column(Text, nullable=True)
    face_registered = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    attendance_records = relationship(
        "Attendance", back_populates="student", cascade="all, delete-orphan"
    )


class Attendance(Base):
    __tablename__ = "attendance"
    __table_args__ = (
        UniqueConstraint("student_id_fk", "date", name="uq_student_date"),
    )

    id = Column(Integer, primary_key=True, index=True)

    # Kept as its own column (not just the FK) so records survive readably
    # even if referential joins are bypassed in a report query.
    student_id_fk = Column(Integer, ForeignKey("students.id"), nullable=False)

    date = Column(Date, nullable=False, index=True)
    time = Column(Time, nullable=False)
    status = Column(String(10), nullable=False)  # "Present" | "Absent"
    method = Column(String(20), nullable=False)  # "AI Face Recognition" | "Manual"
    confidence = Column(Float, nullable=True)  # similarity score, AI method only

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student", back_populates="attendance_records")
