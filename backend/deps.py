# -*- coding: utf-8 -*-
"""
FastAPI dependencies: DB session and the authenticated student.
"""
from __future__ import annotations

from typing import Iterator

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from database import SessionLocal
from models import Student
from security import decode_token

# Sends the standard `Authorization: Bearer <token>` header.
bearer_scheme = HTTPBearer(auto_error=True)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_student(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Student:
    cred_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(creds.credentials)
        student_id = int(payload.get("sub"))
    except (jwt.PyJWTError, TypeError, ValueError):
        raise cred_error

    student = db.get(Student, student_id)
    if student is None or not student.is_active:
        raise cred_error
    return student
