from fastapi import FastAPI
import pickle
from pydantic import BaseModel, Field



with open("wineModel.pkl", "rb") as f:
    model_rf = pickle.load(f)

class wineData(BaseModel):
    alcohol : float
    malic_acid : float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float = Field(alias="od280/od315_of_diluted_wines")
    proline: float

class wineClass(BaseModel):
    Class_type : int

wineapp = FastAPI()
@wineapp.post("/predict", response_model=wineClass)
def predict(wineData : wineData):
    result = model_rf.predict(
        [
            [
                wineData.alcohol,
                wineData.malic_acid,
                wineData.ash,
                wineData.alcalinity_of_ash,
                wineData.magnesium,
                wineData.total_phenols,
                wineData.flavanoids,
                wineData.nonflavanoid_phenols,
                wineData.proanthocyanins,
                wineData.color_intensity,
                wineData.hue,
                wineData.od280_od315_of_diluted_wines,
                wineData.proline
            ]
        ]
    )
    
    return wineClass(Class_type = int(result[0]))



