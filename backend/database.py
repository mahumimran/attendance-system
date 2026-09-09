"""
SQLAlchemy engine, session factory, and declarative base.
All modules import `Base` and `get_db` from here instead of creating
their own engine, so the whole app shares one connection pool.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.utils.config import settings

connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a session, always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't already exist. Called once on startup."""
    from backend.models import models  # noqa: F401  (ensures models are registered)
    Base.metadata.create_all(bind=engine)
