# Deploy on Render

## 1. Add these files to your GitHub repository

- `app.py`
- `requirements.txt`

Commit and push both files to the repository root.

## 2. Create the Render service

1. Sign in to Render and choose **New + → Web Service**.
2. Connect `kartikcodes03/student-lifestyle-clustering`.
3. Select the branch `main`.
4. Choose **Python** as the runtime.
5. Set **Build Command** to:

   ```bash
   pip install -r requirements.txt
   ```

6. Set **Start Command** to:

   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

7. Choose an instance plan and click **Create Web Service**.

## 3. Use the app

After deployment, open the Render URL and upload the CSV file:

`Student Social Media And Mental Health Impact.csv`

The CSV must contain these columns:

- `Stress_Level`
- `Avg_Daily_Usage_Hours`
- `Daily_Unlocks`
- `Sleep_Hours_Per_Night`
- `Physical_Activity_Hours`
- `Study_Hours`
- `Gender`
- `Country`
- `Academic_Level`
- `Most_Used_Platform`
- `Purpose_Of_Use`

The app trains a reproducible Random Forest pipeline from the uploaded data at runtime. It uses the same target definition and engineered features as the supplied notebook, while placing oversampling inside the model pipeline.

## Important

- The app currently retrains when a new dataset is uploaded / app cache is cleared. Large datasets may require a paid Render instance or optimization.
- Do not commit or upload a dataset unless its license permits redistribution.
- Predictions are educational dataset-derived labels, not medical diagnoses.
