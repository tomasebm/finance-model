# ============================================================
# MERVAL DAILY SIGNAL — FULL DEPLOY (FIXED)
# ============================================================
$PROJECT_ID = "merval-482121"
$PROJECT_NUMBER = "176341656154"
$REGION = "us-central1"
$REPO = "merval"
$IMAGE = "daily-signal"
$JOB_NAME = "daily-merval-signal"
$SCHEDULER_NAME = "daily-merval-scheduler"
$SA_EMAIL = "merval-scheduler-sa@merval-482121.iam.gserviceaccount.com"

Write-Host "Authenticating..."
gcloud config set project $PROJECT_ID

# ----------------------------
# ENABLE APIS
# ----------------------------
Write-Host "Enabling APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com cloudscheduler.googleapis.com

# ----------------------------
# CREATE ARTIFACT REGISTRY
# ----------------------------
Write-Host "Creating Artifact Registry..."
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION --quiet 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "Artifact Registry already exists" }

# ----------------------------
# GRANT ALL PERMISSIONS
# ----------------------------
Write-Host "Granting permissions..."

# Cloud Build permissions
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" --role="roles/artifactregistry.writer"

# Service Account permissions
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/run.invoker"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/iam.serviceAccountUser"

# ----------------------------
# BUILD & PUSH IMAGE
# ----------------------------
Write-Host "Building and pushing Docker image..."
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE"

# ----------------------------
# CREATE OR UPDATE CLOUD RUN JOB
# ----------------------------
Write-Host "Creating Cloud Run Job..."
gcloud run jobs create $JOB_NAME --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE" --region $REGION --max-retries 1 --task-timeout 15m --service-account=$SA_EMAIL --set-env-vars="PROJECT_ID=$PROJECT_ID" --quiet 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Job exists, updating..."
    gcloud run jobs update $JOB_NAME --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE" --region $REGION --service-account=$SA_EMAIL --set-env-vars="PROJECT_ID=$PROJECT_ID"
}

# ----------------------------
# TEST JOB
# ----------------------------
Write-Host "Testing job..."
gcloud run jobs execute $JOB_NAME --region $REGION --wait

# ----------------------------
# CREATE SCHEDULER
# ----------------------------
Write-Host "Creating Cloud Scheduler..."
$RUN_URI = "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/${JOB_NAME}:run"

gcloud scheduler jobs create http $SCHEDULER_NAME --location $REGION --schedule "0 21 * * 1-5" --time-zone "America/Argentina/Buenos_Aires" --uri $RUN_URI --http-method POST --oauth-service-account-email $SA_EMAIL --quiet 2>$null

if ($LASTEXITCODE -ne 0) { 
    Write-Host "Scheduler exists, updating..."
    gcloud scheduler jobs update http $SCHEDULER_NAME --location $REGION --schedule "0 21 * * 1-5" --time-zone "America/Argentina/Buenos_Aires" --uri $RUN_URI --http-method POST --oauth-service-account-email $SA_EMAIL
}

Write-Host ""
Write-Host "=========================================="
Write-Host "DEPLOY COMPLETE"
Write-Host "=========================================="
Write-Host "Job: $JOB_NAME"
Write-Host "Schedule: Mon-Fri at 18:00 ART (21:00 UTC)"
Write-Host "Timezone: America/Argentina/Buenos_Aires"
Write-Host "BigQuery: merval_signals.daily_signal"
Write-Host "Service Account: $SA_EMAIL"
Write-Host ""
Write-Host "To view logs:"
Write-Host "gcloud run jobs executions list --job=$JOB_NAME --region=$REGION"
Write-Host ""
Write-Host "To manually trigger:"
Write-Host "gcloud run jobs execute $JOB_NAME --region=$REGION"
Write-Host ""
Write-Host "To check scheduler:"
Write-Host "gcloud scheduler jobs describe $SCHEDULER_NAME --location=$REGION"
Write-Host "=========================================="