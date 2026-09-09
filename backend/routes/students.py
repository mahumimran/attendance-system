from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from backend.database import get_db
from backend.models.models import Student
from backend.schemas.schemas import (
    StudentCreate, StudentUpdate, StudentOut,
    FaceRegisterRequest, FaceRegisterResponse,
)
from backend.services.face_recognition_service import (
    process_registration_frame, FaceQualityError,
)

router = APIRouter(prefix="/students", tags=["Students"])


@router.post("", response_model=StudentOut, status_code=201)
def create_student(payload: StudentCreate, db: Session = Depends(get_db)):
    existing = db.query(Student).filter(Student.student_id == payload.student_id).first()
    if existing:
        raise HTTPException(status_code=409, detail="Student ID is already registered.")

    student = Student(**payload.model_dump())
    db.add(student)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student ID is already registered.")
    db.refresh(student)
    return student


@router.get("", response_model=list[StudentOut])
def list_students(
    class_name: str | None = None,
    section: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Student)
    if class_name:
        query = query.filter(Student.class_name == class_name)
    if section:
        query = query.filter(Student.section == section)
    return query.order_by(Student.name).all()


@router.get("/{student_id}", response_model=StudentOut)
def get_student(student_id: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    return student


@router.put("/{student_id}", response_model=StudentOut)
def update_student(student_id: str, payload: StudentUpdate, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(student, field, value)

    db.commit()
    db.refresh(student)
    return student


@router.delete("/{student_id}", status_code=204)
def delete_student(student_id: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    db.delete(student)
    db.commit()
    return None


@router.post("/{student_id}/face", response_model=FaceRegisterResponse)
def register_face(student_id: str, payload: FaceRegisterRequest, db: Session = Depends(get_db)):
    """
    Runs the full CV pipeline (detect -> validate -> align -> embed) on a
    single captured frame and stores the resulting embedding.
    """
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    try:
        embedding_json, quality_score = process_registration_frame(payload.image_base64)
    except FaceQualityError as e:
        raise HTTPException(status_code=422, detail=str(e))

    student.face_embedding = embedding_json
    student.face_registered = True
    db.commit()

    return FaceRegisterResponse(
        success=True,
        message="Face registered successfully.",
        quality_score=round(quality_score, 1),
    )
