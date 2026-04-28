# MLOps Peer Review: Architecture Assessment

**Reviewer:** Paul Lemoine Vandermoere

---

## Good Practices Checklist Compliance

Based on the course guidelines, here is the assessment of your repository's compliance with standard MLOps development practices:

- [x] **Version Control & Structure:** Git history is clean, and the repository follows a logical, standardized structure (`src/`, `data/`, `notebooks/`).
- [x] **Dependency Management:** Excellent use of tools (`uv`, `pyproject.toml`, `uv.lock`) ensuring strict reproducibility.
- [x] **Security & Configuration:** Secrets and configurations are properly handled (`.env.example` provided, no hardcoded `ACCESS_KEY` in the repository).
- [x] **Data Management:** Raw data and heavy artifacts are excluded from Git (`.gitignore` is properly configured) and fetched externally from S3.
- [x] **Code Quality:** Notebooks were successfully refactored into modular `.py` scripts. CI workflows (GitHub Actions) are implemented for linting (Ruff/PyLint) and testing (Pytest).

---

## General Assessment

Your project demonstrates an high level. The transition from an exploratory NLP notebook to a fully containerized application is interesting. The separation of concerns shows a deep understanding of the ML lifecycle. The extensive README also makes onboarding easy for new developers. Overall, it aligns with the MLOps philosophy taught in the course.

---

## Suggested Areas for Improvement

While the architecture is highly functional, I have identified a few structural points that could be optimized for a more resilient production environment:

### Data URL Management in Docker (`--build-arg` vs `ENV`)
In your current Docker implementation, the S3 URL is passed as a `--build-arg`. 
* **The issue:** This approach bakes the URL directly into the image. If the object storage endpoint changes, the entire Docker image must be rebuilt.
* **Proposal:** Pass this parameter as an environment variable (`ENV`) at runtime (e.g., in your Kubernetes deployment manifests). The Docker image should remain entirely agnostic to environnement.

### Remote MLflow Tracking
You rightfully mentioned this in your "Future Improvements" section. Currently, your MLflow setup relies on a local SQLite backend (`sqlite:///mlflow.db`).
* **The issue:** If the container or pod restarts, the local model registry and experiment history are lost.
* **Proposal:** Configure `train.py` and the FastAPI app to connect to a remote MLflow tracking URI hosted on the SSP Cloud. This will allow your API to dynamically pull the latest champion model without storing artifacts locally inside the container.

### NLP Resources Automation
The manual initialization requires the user to explicitly run `uv run python -m spacy download...`.
* **Proposal:** To eliminate the risk of `RuntimeError` due to missing NLP models, consider adding an automatic verification within `train.py` (using a `try/except` block to download it if missing) or strictly enforcing it within your `install.sh` script.
