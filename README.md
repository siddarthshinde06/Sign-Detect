# SignSense Detector — simple real-time hand sign detection

No data collection tab, no training tab — just a webcam feed that detects
hand signs live using a model you've already trained.

```
signsense-detect/
├── app.py              # FastAPI backend — camera + detection only
├── requirements.txt
├── svm_model.pkl        # <- put YOUR trained model here
├── scaler.pkl            # <- put YOUR trained scaler here
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

## Setup

1. Train your model however you like (your own script, or the earlier
   full SignSense app) so you end up with two files:
   - `svm_model.pkl` — an sklearn `SVC(probability=True)` fitted on
     63-dimensional normalized hand-landmark features
   - `scaler.pkl` — the matching fitted `StandardScaler`

2. Copy both files into this folder, next to `app.py`.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run:
   ```bash
   python app.py
   ```

5. Open **http://localhost:8000**, click **Start Detection**.

If you retrain later, just overwrite `svm_model.pkl` / `scaler.pkl` and
click **Reload Model** in the page — no restart needed.

## Notes

- The webcam opens on the machine running the server, so run this locally.
- Detection uses a 5-frame majority vote and only shows a label once
  average confidence is ≥ 80%; otherwise it shows "Low Confidence".
- If "Start Detection" fails, check that no other app (Zoom, Teams, the
  Windows Camera app, another browser tab) is using the webcam.
