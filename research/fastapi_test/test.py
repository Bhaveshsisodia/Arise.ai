from fastapi import FastAPI , Path , HTTPException ,Query
# from hel_func import load_data
from fastapi.responses import JSONResponse
from pydantic import BaseModel , Field , computed_field
from typing import Annotated , Literal , Optional
import json
app = FastAPI()

class Patient(BaseModel):
    id: Annotated[str, Field(..., description="The unique identifier for the patient", example="P001")]
    name: Annotated[str, Field(..., description="The name of the patient", example="John Doe")]
    city : Annotated[str, Field(..., description="The city where the patient resides", example="New York")]
    age: Annotated[int, Field(...,gt=0 ,lt=120,description="The age of the patient", example=30)]
    gender: Annotated[Literal["male", "female","Others"], Field(..., description="Gender of the Patient")]
    height: Annotated[float , Field(..., gt=0,  description="Height of the patient in meters", example=1.75)]
    weight: Annotated[float, Field(..., gt=0, description="Weight of the patient in kilograms", example=70.5)]

    @computed_field
    @property
    def BMI(self) -> float:
        return round(self.weight / (self.height ** 2), 2)

    @computed_field
    @property
    def verdict(self) -> str:
        if self.BMI < 18.5:
            return "Underweight"
        elif 18.5 <= self.BMI < 24.9:
            return "Normal weight"
        elif 25 <= self.BMI < 29.9:
            return "Overweight"
        else:
            return "Obesity"

class PatientUpdate(BaseModel):
    name: Annotated[Optional[str], Field(default=None)]
    city: Annotated[Optional[str], Field(default=None)]
    age: Annotated[Optional[int], Field(default=None, gt=0)]
    gender: Annotated[Optional[Literal['male', 'female']], Field(default=None)]
    height: Annotated[Optional[float], Field(default=None, gt=0)]
    weight: Annotated[Optional[float], Field(default=None, gt=0)]



def load_data():
    with open("patient.json", "r") as file:
        data = json.load(file)
    return data

def save_data(data):
    with open("patient.json", "w") as file:
        json.dump(data, file)


@app.post('/create')

def create_patient(patient: Patient):
    data = load_data()

    if patient.id in data:
        raise HTTPException(status_code=400, detail="Patient with this ID already exists")
    data[patient.id] = patient.model_dump(exclude=['id'])
    save_data(data)
    return JSONResponse(status_code=201, content={'message':"patient Created Successfully"})

@app.get("/")
def home():
    return JSONResponse(status_code=200 , content={"message":"Home Is loading Successful!"})
@app.get("/")
def home():
    return {"Message": "Home Page"}


@app.get("/test")
def test():
    return {"user":"Hi"}

@app.get("/data")
def get_data():
    data = load_data()
    return data

@app.get("/chunk/{chunk_id}")
def get_chunk(chunk_id: str = Path(..., description="The ID of the chunk to retrieve", example="0")):
    data = load_data()
    data = {item["chunk_id"]: item for item in data}

    if f"JUSNL_petition_{chunk_id}" in data:
        return data[f"JUSNL_petition_{chunk_id}"]

    raise HTTPException(status_code=404, detail="Chunk not found")


@app.get("/sort")
def sort_jusnl(sort_by: str = Query(..., description="The field to sort by Chunk id"
                                    , example="chunk_id"), order: str = Query("asc",
                                                                              description="The order of sorting, either 'asc' or 'desc'",
                                                                                example="asc")):
    valid_fields = ["chunk_id", "page_start", "sub_chunk_index", "text"]
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail="Invalid sort field")

    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Invalid sort order")

    data = load_data()
    sorted_data = sorted(data, key=lambda x: x[sort_by], reverse=(order == "desc"))
    return sorted_data

@app.put('/edit/{patient_id}')
def update_patient(patient_id:str, patient_update:PatientUpdate):
    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail="Patient Not Found")

    existing_patient_info=data[patient_id]

    update_patient_info = patient_update.model_dump(exclude_unset=True)

    for key , value in update_patient_info.items():
        existing_patient_info[key] = value

    existing_patient_info['id'] = patient_id

    patient_pydantic_obj = Patient(**existing_patient_info)

    existing_patient_info = patient_pydantic_obj.model_dump(exclude='id')

    data[patient_id] = existing_patient_info
    print(data)

    save_data(data)

    return JSONResponse(status_code=200, content={'message':"Patient Updated"})

@app.delete("/delete/{patient_id}")
def delete_patient(patient_id :str):
    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail="Patient Not Found")

    del data[patient_id]
    save_data(data)

    return JSONResponse(status_code=200, content={'message':"Patient Delete Successfully"})
