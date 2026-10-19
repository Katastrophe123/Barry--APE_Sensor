"""JWT issuing and verification.

Deliberately small: the project is an agent showcase, not an identity provider.
Tokens are HS256-signed with a shared secret taken from the environment.
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.config import Settings


class AuthError(Exception):
    """Raised when credentials or a token cannot be accepted."""

    def __init__(self, message: str = "Could not validate credentials") -> None:
        super().__init__(message)
        self.message = message


def create_access_token(
    subject: str,
    settings: Settings,
    *,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Return ``(token, expires_in_seconds)`` for ``subject``."""
    expires_in = settings.jwt_expire_minutes * 60
    now = datetime.now(tz=UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
        "iss": settings.app_name,
    }
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """Decode and validate a token, raising :class:`AuthError` when invalid."""
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.app_name,
            options={"require": ["exp", "sub", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Invalid token") from exc
    if not payload.get("sub"):
        raise AuthError("Token is missing a subject")
    return payload


def verify_credentials(username: str, password: str, settings: Settings) -> str:
    """Check demo credentials in constant time and return the subject."""
    user_ok = hmac.compare_digest(username, settings.auth_username)
    pass_ok = hmac.compare_digest(password, settings.auth_password)
    if not (user_ok and pass_ok):
        raise AuthError("Incorrect username or password")
    return username
