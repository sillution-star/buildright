# Mom Test Coach — Setup Guide

A B2C/B2B product validation platform powered by RAG retrieval from the actual Mom Test book.

**Stack:** sentence-transformers (free, local embeddings) · Supabase pgvector · Groq LLaMA 3.3 · FastAPI · Render

---

## What You'll Need (all free)

- Python 3.10+ on your laptop
- A free Supabase account → supabase.com
- A free Groq account → console.groq.com
- A free Render account → render.com
- A free GitHub account → github.com

---

## Step 1: Set Up Supabase (5 minutes)

1. Go to **supabase.com** → New project → give it a name → set a DB password → Create
2. Wait ~2 minutes for it to spin up
3. Click **SQL Editor** in the left sidebar
4. Copy the entire contents of `supabase_setup.sql` and paste it in
5. Click **Run**
6. You should see: "Success. No rows returned"

Now get your keys:
- Go to **Settings → API**
- Copy **Project URL** → this is your `SUPABASE_URL`
- Copy **service_role** key (the long one under "Project API keys") → this is your `SUPABASE_SERVICE_KEY`

---

## Step 2: Get Your Groq API Key (2 minutes)

1. Go to **console.groq.com**
2. Sign up (no credit card)
3. Click **API Keys** → **Create API Key**
4. Name it `mom-test-app`
5. Copy it immediately — shown only once

---

## Step 3: Set Up Your Laptop (10 minutes)

Open Terminal (Mac) or Command Prompt (Windows):

```bash
# Clone or download this project, then:
cd momtest

# Create a virtual environment
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

Create your `.env` file:
```bash
# In the backend/ folder, copy the example:
cp backend/.env.example backend/.env
```

Open `backend/.env` in any text editor and fill in:
```
GROQ_API_KEY=gsk_your_actual_key_here
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here
```

---

## Step 4: Ingest the Book (run once, ~5 minutes)

This downloads the embedding model (~80MB, one time) and stores all book chunks in Supabase.

```bash
# From the momtest/ root folder:
python scripts/ingest.py
```

You'll see:
```
📖 Loading Mom Test book...
Book loaded: 30984 words
✂️  Chunking...
Created 89 chunks
Loading embedding model (downloads once ~80MB)...
Generating embeddings for 89 chunks...
☁️  Storing in Supabase...
Inserted chunks 0–49 / 89
Inserted chunks 50–89 / 89
✅ Done! 89 chunks stored in Supabase.
```

Verify: Go to Supabase → **Table Editor** → you should see `mom_test_chunks` with 89 rows.

---

## Step 5: Run Locally (test before deploying)

```bash
# Start the backend:
cd backend
uvicorn main:app --reload --port 8000
```

You'll see:
```
Loading embedding model...
Embedding model ready.
INFO: Uvicorn running on http://127.0.0.1:8000
```

Open `frontend/index.html` directly in your browser (double-click the file). The app talks to `http://localhost:8000`.

Test it — go through all 4 steps. If you see questions being generated, everything works.

---

## Step 6: Deploy to Render (10 minutes, stays free)

### 6a. Push to GitHub

```bash
cd momtest
git init
git add .
git commit -m "initial commit"
```

Go to github.com → New repository → name it `mom-test-coach` → Create.

```bash
git remote add origin https://github.com/YOUR_USERNAME/mom-test-coach.git
git push -u origin main
```

### 6b. Deploy Backend on Render

1. Go to **render.com** → New → **Web Service**
2. Connect your GitHub account → select `mom-test-coach`
3. Settings:
   - **Name:** momtest-backend
   - **Root Directory:** backend
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Click **Advanced → Add Environment Variable** for each:
   - `GROQ_API_KEY` → your Groq key
   - `SUPABASE_URL` → your Supabase URL
   - `SUPABASE_SERVICE_KEY` → your Supabase service key
5. Click **Create Web Service**
6. Wait ~3 minutes to deploy
7. Copy your backend URL: `https://momtest-backend.onrender.com`

### 6c. Update Frontend API URL

Open `frontend/index.html`, find this line:
```javascript
: "";  // same origin when deployed on Render
```

Change it to your actual backend URL:
```javascript
: "https://momtest-backend.onrender.com";
```

Commit and push:
```bash
git add frontend/index.html
git commit -m "update backend url"
git push
```

### 6d. Deploy Frontend on Render

1. Render → New → **Static Site**
2. Connect same repo
3. Settings:
   - **Name:** momtest-frontend
   - **Root Directory:** frontend
   - **Publish Directory:** .
4. Click **Create Static Site**
5. Your app is live at `https://momtest-frontend.onrender.com`

---

## How It Works (the RAG loop)

```
Your product context
        ↓
sentence-transformers converts it to a vector (locally, free)
        ↓
Supabase pgvector finds the 5 most relevant book passages
        ↓
Those passages + your context → sent to Groq LLaMA 3.3
        ↓
LLM generates questions/report GROUNDED in the actual book text
        ↓
Frontend shows you book excerpts used (transparent)
```

---

## File Structure

```
momtest/
├── backend/
│   ├── main.py          ← FastAPI app (RAG + Groq calls)
│   ├── mom-test.md      ← The book (already here)
│   ├── requirements.txt
│   └── .env.example     ← Copy to .env and fill in keys
├── frontend/
│   └── index.html       ← Complete UI (no build step)
├── scripts/
│   └── ingest.py        ← Run once to chunk + embed the book
├── supabase_setup.sql   ← Run once in Supabase SQL editor
├── render.yaml          ← Render deployment config
└── README.md            ← This file
```

---

## Troubleshooting

**`ingest.py` fails with "sentence_transformers not found"**
→ Make sure your venv is activated: `source venv/bin/activate`

**Backend returns 502 error**
→ Check your GROQ_API_KEY is set correctly in `.env`

**Supabase returns error on chunk insert**
→ Make sure you ran `supabase_setup.sql` first

**Frontend shows "Is the backend running?"**
→ Check backend URL in `index.html` matches your Render backend URL exactly

**Render backend sleeps after 15 minutes (free tier)**
→ First request after sleep takes ~30 seconds. Upgrade to paid ($7/month) or use UptimeRobot to ping it every 10 minutes (free).

---

## What Makes This Different

Every question and every report score is grounded in actual passages from The Mom Test book — not just Claude's general knowledge. The `book_excerpts_used` field in every response shows you exactly which passages were retrieved. This is real RAG, not a system prompt with rules.
