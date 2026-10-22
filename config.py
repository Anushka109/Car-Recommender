import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.urandom(32)
    #SQLALCHEMY_DATABASE_URI = os.getenv(
     #   'DATABASE_URL', 
      #  'postgresql://postgres@localhost:5432/honda_recommender'

    SQLALCHEMY_DATABASE_URI = 'postgresql://humancare02@localhost:5432/honda_recommender'    
    #)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)

