from pydantic import BaseModel
from typing import List

class Landmarks(BaseModel):
    landmarks: List[float]