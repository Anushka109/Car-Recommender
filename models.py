from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class UserResponse(db.Model):
    __tablename__ = 'user_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    gender = db.Column(db.String(20))
    age_group = db.Column(db.String(30))
    work_type = db.Column(db.String(50))
    purposes = db.Column(db.Text)
    seating = db.Column(db.String(20))
    budget = db.Column(db.String(30))
    new_used = db.Column(db.String(10))
    primary_priority = db.Column(db.String(50))
    secondary_priorities = db.Column(db.Text)
    hobbies = db.Column(db.Text)
    
    # Recommendation results
    top_cars = db.Column(db.Text)
    scores = db.Column(db.Text)
    raw_user_features = db.Column(db.Text)  # for future analysis

    def __repr__(self):
        return f"<UserResponse {self.id} - {self.timestamp}>"