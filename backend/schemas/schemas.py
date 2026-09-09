"""
Pydantic schemas used for request validation and response serialization.
Keeping these separate from the ORM models lets the API contract evolve
independently of the database schema.
"""
from datetime import date, time, datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


# ---------- Students ----------

class StudentCreate(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=32)
    name: str = Field(..., min_length=1, max_length=120)
    class_name: str = Field(..., min_length=1, max_length=50)
    section: str = Field(..., min_length=1, max_length=10)
    roll_number: str = Field(..., min_length=1, max_length=20)


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    class_name: Optional[str] = None
    section: Optional[str] = None
    roll_number: Optional[str] = None


class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: str
    name: str
    class_name: str
    section: str
    roll_number: str
    face_registered: bool
    created_at: datetime


class FaceRegisterRequest(BaseModel):
    # base64-encoded JPEG/PNG frame from the browser webcam
    image_base64: str


class FaceRegisterResponse(BaseModel):
    success: bool
    message: str
    quality_score: Optional[float] = None


# ---------- Attendance ----------

class RecognizeRequest(BaseModel):
    image_base64: str


class RecognizeResponse(BaseModel):
    recognized: bool
    student_id: Optional[str] = None
    name: Optional[str] = None
    time: Optional[str] = None
    confidence: Optional[float] = None
    already_marked: bool = False
    message: str


class ManualAttendanceEntry(BaseModel):
    student_id: str
    status: str  # "Present" | "Absent"


class ManualAttendanceRequest(BaseModel):
    date: date
    class_name: str
    section: str
    entries: List[ManualAttendanceEntry]


class AttendanceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date
    time: time
    status: str
    method: str
    confidence: Optional[float] = None
    student: StudentOut


class AttendanceFilter(BaseModel):
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    student_id: Optional[str] = None
    class_name: Optional[str] = None
    section: Optional[str] = None
    status: Optional[str] = None
    method: Optional[str] = None


# ---------- Reports ----------

class ReportRequest(BaseModel):
    report_type: str  # "daily" | "weekly" | "monthly" | "student" | "class"
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    student_id: Optional[str] = None
    class_name: Optional[str] = None
    section: Optional[str] = None
