"""
Helper parsing query-param untuk endpoint list (alerts, predictions).
Nilai tidak valid -> BadParam -> 400 RFC7807 (api.md §Conventions), bukan
diabaikan diam-diam. Nilai boleh diulang (?a=1&a=2) atau dipisah koma.
"""
import uuid
from datetime import timezone as dt_timezone

from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework.response import Response


class BadParam(Exception):
    pass


def problem_400(detail):
    return Response(
        {"type": "about:blank", "title": "Bad request", "status": 400, "detail": detail},
        status=400,
    )


def multi(params, name):
    values = []
    for raw in params.getlist(name):
        values.extend(v.strip() for v in raw.split(",") if v.strip())
    return values


def parse_enum(params, name, allowed):
    """Return list nilai (kosong = tidak difilter)."""
    values = multi(params, name)
    bad = [v for v in values if v not in allowed]
    if bad:
        raise BadParam(f"Nilai '{name}' tidak valid: {bad}. Pilihan: {sorted(allowed)}.")
    return values


def parse_uuid_list(params, name):
    out = []
    for v in multi(params, name):
        try:
            out.append(uuid.UUID(v))
        except ValueError:
            raise BadParam(f"'{name}' bukan UUID valid: {v!r}.")
    return out


def parse_bool(params, name):
    raw = params.get(name)
    if raw is None or raw == "":
        return None
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    raise BadParam(f"'{name}' harus 'true' atau 'false', bukan {raw!r}.")


def parse_dt(params, name):
    raw = params.get(name)
    if not raw:
        return None
    try:
        dt = parse_datetime(raw)
    except ValueError:
        dt = None
    if dt is None:
        raise BadParam(
            f"'{name}' bukan ISO8601 valid: {raw!r}. Gunakan format seperti "
            "2026-09-26T00:00:00Z (tanda '+' di URL harus ditulis %2B)."
        )
    if timezone.is_naive(dt):
        dt = dt.replace(tzinfo=dt_timezone.utc)
    return dt
