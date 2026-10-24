"""
Exception DRF khusus yang dipetakan exception handler RFC7807 (config/exception_handler.py).
Endpoint baru cukup melempar ini — tidak menulis dict problem manual.
"""
from rest_framework.exceptions import APIException


class InvalidQueryParam(APIException):
    status_code = 400
    default_detail = "Parameter query tidak valid."
    default_code = "invalid_query_param"


class Conflict(APIException):
    status_code = 409
    default_detail = "Konflik dengan keadaan sumber daya saat ini."
    default_code = "conflict"


class ServiceUnavailable(APIException):
    status_code = 503
    default_detail = "Layanan sementara tidak tersedia."
    default_code = "service_unavailable"
