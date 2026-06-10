# Deepfake Detection — Django Application

## Requirements
- Python >= 3.6
- Django >= 3.0
- CUDA >= 10.0 (Nvidia GPU mandatory)
- GPU Compute Capability > 3.0

## Directory Structure
- `ml_app` — Core app logic (views.py)
- `project_settings` — Django settings and production configs
- `static` — CSS, JS and JSON files (face-api)
- `templates` — HTML template files

> Before running, create these directories in the project root: `models`, `uploaded_images`, `uploaded_videos`

---

## Running Locally

**Step 1:** Clone the repo
```
git clone https://github.com/hbhandari0005/Deepfake-Detection
```

**Step 2:** Create and activate a virtual environment *(optional)*
```
python -m venv venv
venv\Scripts\activate
```

**Step 3:** Install requirements
```
pip install -r requirements.txt
```

**Step 4:** Train model

> Model filename must follow the format: `model.pt` — the frame count must appear after the 3rd underscore.

**Step 5:** Run the server
```
python manage.py runserver
```