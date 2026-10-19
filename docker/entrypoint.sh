#!/usr/bin/env sh
# Bring the container up with no manual steps: wait for the database, migrate,
# train a first model if none exists, then serve.
set -eu

echo "[entrypoint] waiting for database..."
python - <<'PY'
import asyncio
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

URL = os.environ["DATABASE_URL"]
ATTEMPTS = int(os.environ.get("DB_WAIT_ATTEMPTS", "30"))
DELAY = float(os.environ.get("DB_WAIT_DELAY", "1.0"))


async def main() -> None:
    for attempt in range(1, ATTEMPTS + 1):
        engine = create_async_engine(URL)
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            print(f"[entrypoint] database reachable after {attempt} attempt(s)")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[entrypoint] attempt {attempt}/{ATTEMPTS} failed: {exc}")
            await asyncio.sleep(DELAY)
        finally:
            await engine.dispose()
    print("[entrypoint] database never became reachable", file=sys.stderr)
    raise SystemExit(1)


asyncio.run(main())
PY

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "[entrypoint] running alembic migrations..."
  alembic upgrade head
else
  echo "[entrypoint] RUN_MIGRATIONS=false, skipping migrations"
fi

ARTIFACTS="${ARTIFACTS_DIR:-/app/artifacts}"
mkdir -p "$ARTIFACTS" "${SCHEDULED_INPUT_DIR:-/app/data/incoming}" \
         "${SCHEDULED_OUTPUT_DIR:-/app/data/scored}"

if [ "${TRAIN_ON_START:-false}" = "true" ]; then
  # `current.json` is the registry's pointer; its presence means a model exists.
  if [ -f "$ARTIFACTS/current.json" ]; then
    echo "[entrypoint] a model already exists in $ARTIFACTS, skipping training"
  else
    echo "[entrypoint] no model found; training one on the sample dataset..."
    python scripts/train.py --artifacts "$ARTIFACTS" \
      || echo "[entrypoint] training failed; the API will report model_loaded=false"
  fi
fi

echo "[entrypoint] starting: $*"
exec "$@"
