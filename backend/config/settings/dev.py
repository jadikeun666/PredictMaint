from .base import *  # noqa: F401,F403

DEBUG = True

ALLOWED_HOSTS = ["*"]

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key")
