from pydantic import BaseModel, Field
import datetime

class Auto(BaseModel):
    brand : str
    model : str
    model_year: int = Field(..., gt=1973, lh=datetime.datetime.now().year)
    milage : float = Field(..., gh=0)
    fuel_type: str
    engine: str
    transmission : str
    ext_col : str
    int_col : str
    accident: str
    clean_title: str
