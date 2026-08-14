# -*- coding: utf-8 -*-
"""
Authentication & profile.

Screens: Onboarding -> Register / Login / Forgot password, and the profile
header on the Dashboard.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from deps import get_current_student, get_db
from models import Student
from schemas import (
    ForgotPasswordIn, LoginIn, Message, RegisterIn, ResetPasswordIn,
    StudentOut, TokenOut, UpdateMeIn,
)
from security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(student: Student) -> TokenOut:
    return TokenOut(
        access_token=create_access_token(student.id),
        student=StudentOut.model_validate(student),
    )


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    exists = db.scalar(select(Student).where(Student.email == body.email))
    if exists is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered.")
    student = Student(
        full_name=body.full_name,
        email=body.email,
        password_hash=hash_password(body.password),
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return _token_response(student)


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    student = db.scalar(select(Student).where(Student.email == body.email))
    if student is None or not verify_password(body.password, student.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password.")
    if not student.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account disabled.")
    db.commit()
    return _token_response(student)


@router.post("/logout", response_model=Message)
def logout(student: Student = Depends(get_current_student)):
    # Tokens are stateless JWTs, so there is nothing to invalidate server-side:
    # the client drops its stored token and the old one expires on its own.
    # Requiring a valid token here keeps the endpoint honest about who logged out.
    return Message(message="تم تسجيل الخروج بنجاح.")


@router.post("/forgot-password", response_model=Message)
def forgot_password(body: ForgotPasswordIn, db: Session = Depends(get_db)):
    # Always return 200 so the endpoint can't be used to probe which emails
    # exist. TODO: generate a reset code and email it to the student.
    return Message(message="إذا كان البريد مسجلاً، فسيتم إرسال رمز الاستعادة إليه.")


@router.post("/reset-password", response_model=Message)
def reset_password(body: ResetPasswordIn, db: Session = Depends(get_db)):
    # TODO: verify `body.code` against the stored reset code before allowing this.
    student = db.scalar(select(Student).where(Student.email == body.email))
    if student is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Account not found.")
    student.password_hash = hash_password(body.new_password)
    db.commit()
    return Message(message="تم تحديث كلمة المرور بنجاح.")


@router.get("/me", response_model=StudentOut)
def me(student: Student = Depends(get_current_student)):
    return student


@router.patch("/me", response_model=StudentOut)
def update_me(
    body: UpdateMeIn,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    if body.full_name is not None:
        student.full_name = body.full_name
    if body.current_level is not None:
        student.current_level = body.current_level
    db.commit()
    db.refresh(student)
    return student
