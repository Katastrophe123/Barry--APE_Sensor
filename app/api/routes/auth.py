"""Token issuing endpoint.

Single operator credential pair from the environment, deliberately: the subject
of this project is the prediction service, not identity management. See README
"Limitations".
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import SettingsDep
from app.api.errors import UnauthorizedError
from app.core.security import AuthError, create_access_token, verify_credentials
from app.logging_config import get_logger
from app.models.schemas import TokenRequest, TokenResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Exchange credentials for a JWT",
    responses={401: {"description": "Invalid credentials"}},
)
async def issue_token(payload: TokenRequest, settings: SettingsDep) -> TokenResponse:
    try:
        subject = verify_credentials(payload.username, payload.password, settings)
    except AuthError as exc:
        logger.warning("auth_failed", username=payload.username)
        raise UnauthorizedError(exc.message) from exc

    token, expires_in = create_access_token(subject, settings)
    logger.info("auth_token_issued", subject=subject, expires_in=expires_in)
    return TokenResponse(access_token=token, expires_in=expires_in)
