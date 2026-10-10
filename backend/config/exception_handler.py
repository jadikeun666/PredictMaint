"""
Exception handler RFC 7807 project-wide (api.md §Conventions).

Semua error DRF dibentuk ulang menjadi:
    {"type": "about:blank", "title": <frasa HTTP>, "status": <int>,
     "detail": <teks ringkas>, "code": <kode mesin-terbaca>}
plus `errors` ({field: [pesan, ...]}) hanya untuk 400 validasi serializer.
`code` dan `errors` adalah member ekstensi (diizinkan RFC 7807).

- Header respons asli (WWW-Authenticate, Retry-After, ...) dipertahankan,
  karena yang diganti hanya body respons yang dibuat handler bawaan DRF.
- Exception tak terduga -> 500 generik; traceback dicatat di logger
  `django.request`, TIDAK dikirim ke klien.
- Content-Type `application/problem+json` hanya jika renderer yang
  dinegosiasikan adalah JSON (agar Browsable API HTML tidak rusak).
- `api_not_found` = handler404 untuk path /api/ (hanya aktif saat DEBUG=False;
  Django memakai halaman 404 teknis saat DEBUG=True). Path lain -> bawaan Django.
"""
import http
import logging

from django.core.exceptions import PermissionDenied as DjangoPermissionDenied
from django.http import Http404, JsonResponse
from django.views.defaults import page_not_found
from rest_framework import exceptions
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler
from rest_framework.views import set_rollback

logger = logging.getLogger("django.request")

PROBLEM_CONTENT_TYPE = "application/problem+json"
MAX_SUMMARY_ITEMS = 5


def _title(status):
    try:
        return http.HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def problem_body(status, detail, code, errors=None):
    body = {
        "type": "about:blank",
        "title": _title(status),
        "status": status,
        "detail": detail,
        "code": code,
    }
    if errors is not None:
        body["errors"] = errors
    return body


def _plain(node):
    """ErrorDetail/ReturnDict/dst -> struktur JSON polos."""
    if isinstance(node, dict):
        return {str(k): _plain(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(v) for v in node]
    return str(node)


def _flatten(node, prefix=""):
    """Hasilkan pasangan (path, pesan) dari struktur error bertingkat."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            if isinstance(v, (dict, list, tuple)):
                yield from _flatten(v, f"{prefix}[{i}]" if prefix else f"[{i}]")
            else:
                yield prefix, str(v)
    else:
        yield prefix, str(node)


def _summary(errors):
    items = list(_flatten(errors))
    parts = [f"{path}: {msg}" if path else msg for path, msg in items[:MAX_SUMMARY_ITEMS]]
    text = "; ".join(parts)
    if len(items) > MAX_SUMMARY_ITEMS:
        text += f" (+{len(items) - MAX_SUMMARY_ITEMS} lainnya)"
    return f"Data tidak valid: {text}"


def _set_problem_content_type(response, context):
    request = context.get("request") if context else None
    if isinstance(getattr(request, "accepted_renderer", None), JSONRenderer):
        response.content_type = PROBLEM_CONTENT_TYPE


def _server_error(exc, context):
    set_rollback()
    request = context.get("request") if context else None
    view = context.get("view") if context else None
    logger.error(
        "Internal Server Error (unhandled exception in %s)",
        view.__class__.__name__ if view is not None else "?",
        exc_info=exc,
        extra={"status_code": 500, "request": getattr(request, "_request", request)},
    )
    response = Response(
        problem_body(500, "Terjadi kesalahan internal pada server.", "server_error"),
        status=500,
    )
    _set_problem_content_type(response, context)
    return response


def problem_exception_handler(exc, context):
    response = drf_exception_handler(exc, context)
    if response is None:  # bukan APIException/Http404/PermissionDenied -> 500
        return _server_error(exc, context)

    # Samakan dengan konversi di handler bawaan DRF agar bisa membaca detail/kode.
    if isinstance(exc, Http404):
        exc = exceptions.NotFound(*(exc.args))
    elif isinstance(exc, DjangoPermissionDenied):
        exc = exceptions.PermissionDenied(*(exc.args))

    status = response.status_code
    if isinstance(exc, exceptions.ValidationError):
        errors = _plain(exc.detail)
        if not isinstance(errors, dict):
            errors = {"non_field_errors": errors if isinstance(errors, list) else [errors]}
        body = problem_body(status, _summary(errors), "validation_error", errors)
    else:
        detail_obj = exc.detail
        code = None
        if isinstance(detail_obj, dict) and "detail" in detail_obj:
            # gaya simplejwt: {"detail": ..., "code": ..., "messages": [...]}
            detail = str(detail_obj["detail"])
            code = str(detail_obj.get("code") or "") or None
        elif isinstance(detail_obj, (dict, list)):
            detail = "; ".join(m for _, m in _flatten(detail_obj)) or str(exc.default_detail)
        else:
            detail = str(detail_obj)
        if not code:
            codes = exc.get_codes()
            code = codes if isinstance(codes, str) else exc.default_code
        body = problem_body(status, detail, code)

    response.data = body
    _set_problem_content_type(response, context)
    return response


def api_not_found(request, exception, template_name="404.html"):
    """handler404: JSON RFC7807 untuk /api/, halaman bawaan Django untuk path lain."""
    if request.path.startswith("/api/"):
        return JsonResponse(
            problem_body(404, "Endpoint tidak ditemukan.", "not_found"),
            status=404,
            content_type=PROBLEM_CONTENT_TYPE,
        )
    return page_not_found(request, exception)
