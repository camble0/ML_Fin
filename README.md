# Rwanda Financial Vulnerability Intelligence

Streamlit application for the NISR 2026 Big Data Hackathon. It separates EICV7 survey observations, saved-model estimates, and proposed responses. The public deployment serves precomputed one-dimension aggregates; it does not serve household-level rows or raw survey files.

## Runtime and entry point

The Streamlit Community Cloud entry point is `app.py` at the repository root. Runtime dependencies are pinned in `requirements.txt` for Python 3.12 and match the versions used to serialize the saved scikit-learn model.

The public app requires these committed runtime assets:

- `data/public/household_aggregates.json`
- `data/public/analytical_report.json`
- `data/processed/eicv7_priority_structure.json`
- `data/processed/eicv7_metadata_inventory.json`
- `models/financial_inclusion_baseline/`

The household-level CSV, Parquet file, raw `.dta` survey files, and local-only reports are intentionally excluded from Git. No synthetic data fallback is used. If a public data asset is missing, the app explains that analysis is unavailable; if a model artifact is missing, prediction is unavailable while aggregate analysis remains usable.

## Local setup

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The app opens at `http://localhost:8501`.

## Prepare public assets

The checked-in `data/public/` files are derived from the processed EICV7 household dataset. To regenerate them, place an authorized `financial_inclusion_household.csv` at `data/processed/financial_inclusion_household.csv` and ensure the analytical report and code-label structure files are present. Then run:

```bash
python src/data/prepare_public_assets.py
```

The script publishes only separate province, district, residence-area, and consumption-quintile aggregates; groups under 50 resolved households are suppressed. It does not emit household or cluster identifiers and removes the held-out demo household ID from the public report. This is a risk-reduction measure, not a guarantee of anonymity. Confirm the dataset license, disclosure rules, and authorization for public release before publishing the output. Do not copy raw survey data or the household-level CSV into the deployment repository.

Rebuilding the dataset from raw Stata inputs is a separate, authorized local workflow and is not required to run the public app. Install `requirements-dev.txt` for the extra `pyreadstat` package before running the dataset builder or test suite. Keep raw inputs outside the public repository.

## Publish on Streamlit Community Cloud

1. Create an empty GitHub repository, then connect this local repository (replace the placeholders with its URL):

	```bash
	git remote add origin https://github.com/<YOUR-USERNAME>/<REPOSITORY>.git
	git branch -M main
	```

	Include the five runtime asset paths listed above, `app.py`, `src/`, `requirements.txt`, and this README. Do not force-add ignored raw data, household-level files, `.env` files, or `.streamlit/secrets.toml`.
2. Before pushing, inspect exactly what will be published:

	```bash
	git status --short
	git diff --check
	git add app.py src tests requirements.txt requirements-dev.txt README.md .gitignore data/public data/processed/eicv7_priority_structure.json data/processed/eicv7_metadata_inventory.json models/financial_inclusion_baseline
	git diff --cached --stat
	git diff --cached --name-only
	```

	Confirm the staged file list contains no raw `.dta`, household-level CSV/Parquet, credentials, or personal records. Commit and push only after that review:

	```bash
	git commit -m "Prepare Streamlit Cloud deployment"
	git push -u origin main
	```

3. Sign in to Streamlit Community Cloud, choose **Create app**, select the GitHub repository and branch, set the main file path to `app.py`, and choose Python 3.12 in advanced settings.
4. Deploy. Community Cloud installs the pinned packages from the root `requirements.txt`; the public app URL can then be shared with the judges. Check the deployment logs if the build fails and verify Overview, Explore, Predict, and Intervention intelligence on the hosted URL.

No API keys, credentials, environment variables, or Streamlit secrets are needed. Do not add secrets unless a future integration requires them; configure any such values in Community Cloud's app Secrets settings and read them through `st.secrets`, never commit them to GitHub.

## Model and interpretation limits

The provisional target is whether a household has a bank or Mobile Money account, derived from EICV7 `s10cq4`. Unanswered `s10cq4` records with exclusively No answers to `s10cq1` are recorded as structural No with provenance; confirm this skip interpretation against the questionnaire before treating it as final. Credit and service-access features may be contemporaneous proxies, not causes.

Model metrics are from one held-out `StratifiedGroupKFold` split by `clust` and are not survey-weighted. Predictions are not calibrated or causal, and group comparisons are descriptive rather than official poverty measures or intervention impact estimates. FinScope 2024 is not merged because survey compatibility has not been established.

## Tests

Install the extra local test/data-preparation dependency first:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```
