from sqlalchemy import Column, Integer, Float, String, ForeignKey, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    age = Column(Float)
    gender = Column(Integer)
    sleep_duration = Column(Float)
    sleep_quality = Column(Integer)
    bmi_category = Column(String)
    stress_level = Column(Integer)
    physical_activity_duration = Column(Float)
    steps_per_day = Column(Integer)
    prediction = Column(JSON)
    predicted_class = Column(String)
    confidence = Column(Float)