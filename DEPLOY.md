# Deployment Guide

## Deploy to Render (Recommended - Free)

### 1. Create a Render Account
- Go to https://render.com
- Sign up with GitHub

### 2. Prepare Your Repository
Ensure these files are in your GitHub repo:
```
├── src/
│   ├── web_app.py
│   ├── alerting_logic.py
│   └── templates/
│       └── index.html
├── requirements.txt
└── render.yaml (create this)
```

### 3. Create render.yaml
```yaml
services:
  - type: web
    name: network-traffic-intelligence
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn src.web_app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: FLASK_ENV
        value: production
```

### 4. Update requirements.txt
Add to requirements.txt:
```
flask
gunicorn
numpy
```

### 5. Update web_app.py for Production
Add this at the bottom of web_app.py:
```python
# For production
import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

### 6. Deploy
1. In Render Dashboard, click "New +" → "Web Service"
2. Connect your GitHub repository
3. Render will auto-detect the configuration
4. Click "Deploy"

Your app will be live at: `https://network-traffic-intelligence.onrender.com`

---

## Deploy to Heroku (Alternative)

### 1. Install Heroku CLI
```powershell
# Windows (PowerShell)
winget install Heroku.HerokuCLI
```

### 2. Create Heroku App
```powershell
heroku login
heroku create network-traffic-intelligence
```

### 3. Create Procfile
Create file named `Procfile` (no extension):
```
web: gunicorn src.web_app:app
```

### 4. Create runtime.txt
```
python-3.11.0
```

### 5. Deploy
```powershell
git add .
git commit -m "Add web interface"
git push heroku main
```

---

## Deploy to PythonAnywhere (Free, Easiest)

### 1. Sign Up
- Go to https://www.pythonanywhere.com
- Create free account

### 2. Upload Files
- Go to Files tab
- Upload your project files

### 3. Create Web App
- Go to Web tab
- Click "Add a new web app"
- Choose "Flask" and Python 3.11

### 4. Configure
- Set working directory to your project folder
- Update WSGI file to point to `src/web_app.py`
- Reload the web app

---

## Local Testing

```powershell
# Install dependencies
pip install flask numpy gunicorn

# Run locally
python src/web_app.py

# Open browser to http://localhost:5000
```

---

## Environment Variables

Create `.env` file for local development:
```
FLASK_ENV=development
PORT=5000
HARDWARE_CAPACITY=45
```

For production, set these in your hosting platform's dashboard.
