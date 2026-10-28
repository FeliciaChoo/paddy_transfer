# Publishing the dashboard so your supervisor can open it

Recommended: **Streamlit Community Cloud**, from a **private** GitHub repository, shared with your supervisor's email.

## Before you publish (checklist)
1. Fill in your own details in `dashboard/about_config.py` (the lines in brackets show on the Overview page).
2. Check you may share the data. `data/*.csv` goes into the repository: it holds the yield figures per granary. If the yield data came from an
   agency with conditions, keep the repository **private** and the app **private**.
3. Run it once on your computer: `streamlit run app.py`.
4. Do NOT commit `.streamlit/secrets.toml` (the `.gitignore` already blocks it). The AI note is optional: without a key the app still works.

## Steps
1. Create a free GitHub account and a **new private repository** (for example `acts-granary-monitor`).
2. In PowerShell, in the project folder:
   ```powershell
   git init
   git add .
   git status
   ```
   Read the `git status` list. It should show `app.py`, `dashboard/`, `acts_transfer/`, `pipeline/`, `tests/`, `docs/`, `geo/`, `external/`, `data/*.csv`, `requirements.txt`, `README.md`.
   It must NOT show `.venv`, `secrets.toml`, `DailyGranary*.csv`, `eda/`, `final_model/` or `gee_cache/`. Then:
   ```powershell
   git commit -m "
 Granary Monitor"
   git branch -M main
   git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**, pick the repository, branch `main`, and main file `app.py`.
   Under **Advanced settings > Secrets** you may paste `GEMINI_API_KEY = "your-key"` (optional). Press **Deploy** and wait for the first build.
4. Share it. An app deployed from a private repository is private. In the app, press **Share**, enter your supervisor's email and press **Invite**
   (or use App settings > Sharing > "Only specific people can view this app"). They get a link by email and sign in with Google or an emailed link.
   Community Cloud allows **one private app at a time**.
5. To update later: change files, then `git add .`, `git commit -m "update"`, `git push`. The app redeploys by itself.

## Things to know
- **Decisions saved in the Decision tracker are stored in a file on the server and are cleared when the app restarts.** For a demo that is fine.
  For permanent storage set `DATABASE_URL` in Secrets to a PostgreSQL or MySQL database (see README, section Database).
- **The data shown ends in December 2024.** Say so when you share it, or refresh the extraction first.
- **Making the app public** lets anyone with the link press Save on the tracker. Keep it private while the data is not cleared for sharing.
- Quick but temporary alternative: run `streamlit run app.py` on your computer and expose it with a tunnelling tool (for example ngrok). It only works
  while your computer is on.
