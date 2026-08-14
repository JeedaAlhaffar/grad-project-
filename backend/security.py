# -*- coding: utf-8 -*-
"""
Password hashing and JWT helpers.

Passwords are hashed with bcrypt (via passlib). Auth uses stateless JWT
bearer tokens, which is the simplest thing for a Flutter client to store and
send back in the `Authorization: Bearer <token>` header.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt  # PyJWT

from config import settings

# bcrypt only hashes the first 72 bytes; truncate explicitly to avoid the
# "password cannot be longer than 72 bytes" error on bcrypt >= 4.
_BCRYPT_MAX = 72


# --- passwords ---------------------------------------------------------------
def hash_password(plain: str) -> str:
    digest = bcrypt.hashpw(plain.encode("utf-8")[:_BCRYPT_MAX], bcrypt.gensalt())
    return digest.decode("utf-8")


def verify_password(plain: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:_BCRYPT_MAX], hashed.encode("utf-8"))
    except ValueError:
        return False


# --- JWT ---------------------------------------------------------------------
def create_access_token(subject: str | int, expires_minutes: int | None = None) -> str:
    """Create a signed access token whose `sub` claim is the student id."""
    minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(subject),
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode/verify a token. Raises jwt.PyJWTError on any problem."""
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
