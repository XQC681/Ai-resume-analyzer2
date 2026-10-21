# AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get an ATS score, a score breakdown,
missing keywords and specific improvement suggestions, powered by Google Gemini Flash.
Optionally paste a job description for a tailored keyword match.

## Features
- Overall ATS score (0-100) computed from five weighted categories
- Strengths, missing keywords, prioritized improvements and example rewrites
- Optional job description matching
- Downloadable JSON report

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Get a free API key at https://aistudio.google.com and paste it in the sidebar,
or set it as an environment variable: `GEMINI_API_KEY`.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (app.py, requirements.txt, README.md).
2. Go to https://share.streamlit.io and click **Create app**.
3. Choose your repo, branch `main`, main file `app.py`.
4. Under **Advanced settings > Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes
- Default model is `gemini-2.5-flash`; change it in the sidebar if you want another Gemini model.
- Scanned/image-only PDFs have no readable text, so they can't be analyzed (ATS systems can't read them either).
- Resumes are sent to the Gemini API. Don't upload anything you aren't comfortable sharing.
- Scores are AI estimates, not the output of any real employer's ATS.
