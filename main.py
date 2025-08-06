from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()

# Set up templates folder (and static if needed)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predictdata", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predictdata", response_class=HTMLResponse)
async def predict_data(
    request: Request,
    gender: str = Form(...),
    ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    try:
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=writing_score,
            writing_score=reading_score
        )

        pred_df = data.get_data_as_data_frame()

        print("📥 DataFrame:")
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        print("✅ Prediction result:", results)

        return templates.TemplateResponse("home.html", {
            "request": request,
            "results": results[0]
        })

    except Exception as e:
        import traceback
        error_str = ''.join(traceback.format_exception_only(type(e), e))
        print("❌ Exception Traceback:\n", error_str)

        return templates.TemplateResponse("home.html", {
            "request": request,
            "results": f"❌ Internal Error: {error_str}"
        })

