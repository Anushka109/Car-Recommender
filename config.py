import os
from datetime import timedelta


def get_database_uri():
    # Prefer Render / production env var when provided.
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    # Fallback to SQLite for local/dev and simple deployed apps.
    # This avoids the localhost:5432 Postgres dependency that caused Render deploys to fail.
    return "sqlite:///car_recommender.db"


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = get_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
