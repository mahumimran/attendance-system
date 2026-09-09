from datetime import datetime, date as date_type

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.models import Student, Attendance
from backend.schemas.schemas import (
    RecognizeRequest, RecognizeResponse,
    ManualAttendanceRequest, AttendanceOut,
)
from backend.services.face_recognition_service import (
    process_recognition_frame, embedding_from_json, FaceQualityError,
)
from backend.services.attendance_service import mark_attendance, get_existing_attendance

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.post("/recognize", response_model=RecognizeResponse)
def recognize_attendance(payload: RecognizeRequest, db: Session = Depends(get_db)):
    """
    Called repeatedly (e.g. every ~1.5s) by the live camera view.
    Runs detection+embedding on the incoming frame, compares against every
    registered student's stored embedding, and marks attendance on a
    confident match.
    """
    registered = (
        db.query(Student)
        .filter(Student.face_registered.is_(True), Student.face_embedding.isnot(None))
        .all()
    )
    if not registered:
        raise HTTPException(
            status_code=400,
            detail="No students with registered faces yet. Register students first.",
        )

    known_embeddings = [embedding_from_json(s.face_embedding) for s in registered]
    known_ids = [s.id for s in registered]

    try:
        result = process_recognition_frame(payload.image_base64, known_embeddings, known_ids)
    except FaceQualityError as e:
        # No/multiple/low-quality faces are reported as recognized=False
        # with the specific reason, not a hard 500 error.
        return RecognizeResponse(recognized=False, message=str(e))

    if not result["matched"]:
        return RecognizeResponse(recognized=False, message="Unknown Student")

    student = db.query(Student).filter(Student.id == result["student_pk"]).first()

    record, created, message = mark_attendance(
        db,
        student_pk=student.id,
        status="Present",
        method="AI Face Recognition",
        confidence=result["confidence"],
    )

    return RecognizeResponse(
        recognized=True,
        student_id=student.student_id,
        name=student.name,
        time=record.time.strftime("%I:%M %p"),
        confidence=result["confidence"],
        already_marked=not created,
        message=message,
    )


@router.post("/manual")
def submit_manual_attendance(payload: ManualAttendanceRequest, db: Session = Depends(get_db)):
    results = []
    for entry in payload.entries:
        student = db.query(Student).filter(Student.student_id == entry.student_id).first()
        if not student:
            results.append({"student_id": entry.student_id, "success": False, "message": "Student not found."})
            continue

        record, created, message = mark_attendance(
            db,
            student_pk=student.id,
            status=entry.status,
            method="Manual",
            for_date=payload.date,
        )
        results.append({"student_id": entry.student_id, "success": True, "message": message})

    return {"results": results}


@router.get("", response_model=list[AttendanceOut])
def list_attendance(
    date_from: date_type | None = None,
    date_to: date_type | None = None,
    student_id: str | None = None,
    class_name: str | None = None,
    section: str | None = None,
    status: str | None = None,
    method: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Attendance).join(Student)

    if date_from:
        query = query.filter(Attendance.date >= date_from)
    if date_to:
        query = query.filter(Attendance.date <= date_to)
    if student_id:
        query = query.filter(Student.student_id == student_id)
    if class_name:
        query = query.filter(Student.class_name == class_name)
    if section:
        query = query.filter(Student.section == section)
    if status:
        query = query.filter(Attendance.status == status)
    if method:
        query = query.filter(Attendance.method == method)

    return query.order_by(Attendance.date.desc(), Attendance.time.desc()).all()


@router.get("/{student_id}", response_model=list[AttendanceOut])
def get_student_attendance(student_id: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    return (
        db.query(Attendance)
        .filter(Attendance.student_id_fk == student.id)
        .order_by(Attendance.date.desc())
        .all()
    )


@router.get("/status/today-summary")
def today_summary(db: Session = Depends(get_db)):
    """Powers the dashboard cards: total / present / absent / percentage."""
    today = datetime.now().date()
    total_students = db.query(Student).count()
    present_today = (
        db.query(Attendance)
        .filter(Attendance.date == today, Attendance.status == "Present")
        .count()
    )
    absent_today = max(total_students - present_today, 0)
    percentage = round((present_today / total_students) * 100, 1) if total_students else 0.0

    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_percentage": percentage,
    }


@router.get("/status/weekly-trend")
def weekly_trend(db: Session = Depends(get_db)):
    """Powers the 'Weekly Attendance' chart: present/absent counts, last 7 days."""
    from datetime import timedelta

    total_students = db.query(Student).count()
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]

    labels, present_counts, absent_counts = [], [], []
    for d in days:
        present = (
            db.query(Attendance)
            .filter(Attendance.date == d, Attendance.status == "Present")
            .count()
        )
        labels.append(d.strftime("%a"))
        present_counts.append(present)
        absent_counts.append(max(total_students - present, 0))

    return {"labels": labels, "present": present_counts, "absent": absent_counts}
