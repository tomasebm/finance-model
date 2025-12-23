# ============================================================
# BIGQUERY SETUP FOR MERVAL SIGNALS
# ============================================================
$PROJECT_ID = "merval-482121"
$REGION = "us-central1"
$DATASET = "merval_signals"
$TABLE = "daily_signal"
$SA_NAME = "merval-scheduler-sa"
$SA_EMAIL = "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

Write-Host "Getting access token..."
$TOKEN = gcloud auth application-default print-access-token

# ----------------------------
# CREATE DATASET
# ----------------------------
Write-Host "Creating BigQuery dataset..."
$DATASET_BODY = @{
    datasetReference = @{
        projectId = $PROJECT_ID
        datasetId = $DATASET
    }
    location = $REGION
} | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Uri "https://bigquery.googleapis.com/bigquery/v2/projects/$PROJECT_ID/datasets" -Method POST -Headers @{Authorization="Bearer $TOKEN"; "Content-Type"="application/json"} -Body $DATASET_BODY
    Write-Host "Dataset created"
} catch {
    if ($_.Exception.Response.StatusCode -eq 409) {
        Write-Host "Dataset already exists"
    } else {
        Write-Host "Error: $($_.Exception.Message)"
    }
}

# ----------------------------
# CREATE TABLE
# ----------------------------
Write-Host "Creating BigQuery table..."
$TABLE_BODY = @{
    tableReference = @{
        projectId = $PROJECT_ID
        datasetId = $DATASET
        tableId = $TABLE
    }
    schema = @{
        fields = @(
            @{name="date"; type="DATE"; mode="REQUIRED"},
            @{name="signal_strength"; type="FLOAT64"},
            @{name="exposure_spy"; type="FLOAT64"},
            @{name="exposure_ggal"; type="FLOAT64"},
            @{name="shock_nextday"; type="INTEGER"},
            @{name="created_at"; type="TIMESTAMP"}
        )
    }
} | ConvertTo-Json -Depth 10

try {
    $result = Invoke-RestMethod -Uri "https://bigquery.googleapis.com/bigquery/v2/projects/$PROJECT_ID/datasets/$DATASET/tables" -Method POST -Headers @{Authorization="Bearer $TOKEN"; "Content-Type"="application/json"} -Body $TABLE_BODY
    Write-Host "Table created"
} catch {
    if ($_.Exception.Response.StatusCode -eq 409) {
        Write-Host "Table already exists"
    } else {
        Write-Host "Error: $($_.Exception.Message)"
    }
}


# ============================================================
# CREATE SERVICE ACCOUNT FIRST
# ============================================================
$PROJECT_ID = "merval-482121"
$SA_NAME = "merval-scheduler-sa"
$SA_EMAIL = "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

Write-Host "Creating Service Account..."
gcloud iam service-accounts create $SA_NAME --display-name "Scheduler for MERVAL daily job" --project=$PROJECT_ID

Write-Host "Granting BigQuery permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/bigquery.dataEditor"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/bigquery.jobUser"

Write-Host ""
Write-Host "Service Account created: $SA_EMAIL"

# ----------------------------
# GRANT PERMISSIONS
# ----------------------------
Write-Host "Granting BigQuery permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/bigquery.dataEditor"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/bigquery.jobUser"

Write-Host ""
Write-Host "BIGQUERY SETUP COMPLETE"
Write-Host "Dataset: $DATASET"
Write-Host "Table: $TABLE"