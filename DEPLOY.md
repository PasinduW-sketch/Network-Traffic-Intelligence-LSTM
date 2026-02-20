# Deployment Guide - 100% FREE Options

## Option 1: PythonAnywhere (RECOMMENDED - Completely Free)

**No sleep, no daily limits, always free tier available!**

### 1. Sign Up
- Go to https://www.pythonanywhere.com
- Create free account (no credit card needed)

### 2. Open Bash Console
- Go to **Consoles** tab
- Click **Bash** to open a terminal

### 3. Clone Your Repository
```bash
git clone https://github.com/PasinduW-sketch/Network-Traffic-Intelligence-LSTM.git
cd Network-Traffic-Intelligence-LSTM
```

### 4. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install flask numpy
```

### 5. Create Web App
- Go to **Web** tab
- Click **Add a new web app**
- Choose **Manual configuration**
- Select **Python 3.11**

### 6. Configure WSGI File
Click on the WSGI configuration file link and replace with:
```python
import sys
path = '/home/yourusername/Network-Traffic-Intelligence-LSTM/src'
if path not in sys.path:
    sys.path.insert(0, path)

from web_app import app as application
```

### 7. Set Working Directory
- In Web tab, set **Working directory** to:
  `/home/yourusername/Network-Traffic-Intelligence-LSTM`

### 8. Reload
Click **Reload** button

Your app will be live at: `https://yourusername.pythonanywhere.com`

---

## Option 2: Vercel (Serverless - Free Forever)

**Good for low-traffic demos**

### 1. Install Vercel CLI
```powershell
npm i -g vercel
```

### 2. Create vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "src/web_app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "src/web_app.py"
    }
  ]
}
```

### 3. Deploy
```powershell
vercel --prod
```

---

## Option 3: Railway (Free Tier)

**$5 credit/month free, no sleep**

### 1. Sign Up
- https://railway.app
- Connect GitHub

### 2. Deploy from Repo
- Click **New Project**
- Select **Deploy from GitHub repo**
- Choose your repository
- Railway auto-detects Python/Flask

---

## Option 4: Always-Local with ngrok (For Demos)

**Free, run on your computer, share via public URL**

### 1. Install ngrok
```powershell
choco install ngrok
# or download from https://ngrok.com
```

### 2. Start Your App
```powershell
python src/web_app.py
```

### 3. Expose to Internet
```powershell
ngrok http 5000
```

You'll get a public URL like `https://abc123.ngrok.io` that tunnels to your local app.

**Pros:** Completely free, no limits
**Cons:** URL changes every time, computer must stay on

---

## Option 5: Glitch (Free, Always On)

**Simple, beginner-friendly**

### 1. Go to https://glitch.com
### 2. Click **New Project** → **Import from GitHub**
### 3. Paste your repo URL
### 4. Glitch auto-runs the app

**Note:** May sleep after 5 mins inactivity, wakes on request

---

## Summary: Best Free Options

| Platform | Always On | Custom Domain | Best For |
|----------|-----------|---------------|----------|
| **PythonAnywhere** | Yes | Yes | Production, always running |
| **Railway** | Yes ($5/mo) | Yes | Small projects |
| **Vercel** | Yes | Yes | Serverless, low traffic |
| **ngrok** | Your PC | No | Quick demos |
| **Glitch** | Wakes on request | No | Learning/testing |

---

## Local Testing

```powershell
pip install flask numpy
python src/web_app.py
# Open http://localhost:5000
```
