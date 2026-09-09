import io
from datetime import datetime, timedelta, date as date_type

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.models import Attendance, Student

router = APIRouter(prefix="/reports", tags=["Reports"])


def _build_dataframe(
    db: Session,
    date_from: date_type | None,
    date_to: date_type | None,
    student_id: str | None,
    class_name: str | None,
    section: str | None,
) -> pd.DataFrame:
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

    rows = query.all()
    data = [
        {
            "Student ID": r.student.student_id,
            "Name": r.student.name,
            "Class": r.student.class_name,
            "Section": r.student.section,
            "Date": r.date.isoformat(),
            "Time": r.time.strftime("%H:%M:%S"),
            "Status": r.status,
            "Method": r.method,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


@router.get("/summary")
def report_summary(
    report_type: str = Query(..., pattern="^(daily|weekly|monthly|student|class)$"),
    date_from: date_type | None = None,
    date_to: date_type | None = None,
    student_id: str | None = None,
    class_name: str | None = None,
    section: str | None = None,
    db: Session = Depends(get_db),
):
    """Returns aggregated attendance-percentage figures for on-screen reporting."""
    today = datetime.now().date()

    if report_type == "daily" and not date_from:
        date_from = date_to = today
    elif report_type == "weekly" and not date_from:
        date_from = today - timedelta(days=today.weekday())
        date_to = today
    elif report_type == "monthly" and not date_from:
        date_from = today.replace(day=1)
        date_to = today

    df = _build_dataframe(db, date_from, date_to, student_id, class_name, section)

    if df.empty:
        return {"total_records": 0, "present": 0, "absent": 0, "attendance_percentage": 0.0, "by_student": []}

    present = int((df["Status"] == "Present").sum())
    absent = int((df["Status"] == "Absent").sum())
    total = present + absent
    percentage = round((present / total) * 100, 1) if total else 0.0

    by_student = (
        df.groupby(["Student ID", "Name"])["Status"]
        .apply(lambda s: round((s == "Present").sum() / len(s) * 100, 1))
        .reset_index(name="Attendance %")
        .to_dict(orient="records")
    )

    return {
        "total_records": total,
        "present": present,
        "absent": absent,
        "attendance_percentage": percentage,
        "by_student": by_student,
    }


@router.get("/export")
def export_report(
    file_format: str = Query("csv", pattern="^(csv|excel)$"),
    date_from: date_type | None = None,
    date_to: date_type | None = None,
    student_id: str | None = None,
    class_name: str | None = None,
    section: str | None = None,
    db: Session = Depends(get_db),
):
    df = _build_dataframe(db, date_from, date_to, student_id, class_name, section)
    buffer = io.BytesIO()

    if file_format == "csv":
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=attendance_report.csv"},
        )

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=attendance_report.xlsx"},
    )
