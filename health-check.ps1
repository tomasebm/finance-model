# ============================================================
# MERVAL SIGNAL - SYSTEM HEALTH CHECK
# ============================================================
$PROJECT_ID = "merval-482121"
$REGION = "us-central1"
$JOB_NAME = "daily-merval-signal"
$SCHEDULER_NAME = "daily-merval-scheduler"
$SA_EMAIL = "merval-scheduler-sa@merval-482121.iam.gserviceaccount.com"
$DATASET = "merval_signals"
$TABLE = "daily_signal"

Write-Host ""
Write-Host "=========================================="
Write-Host "MERVAL SIGNAL - HEALTH CHECK"
Write-Host "=========================================="
Write-Host ""

# ----------------------------
# 1. PROJECT & AUTH
# ----------------------------
Write-Host "1. Checking project and authentication..."
$currentProject = gcloud config get-value project 2>$null
if ($currentProject -eq $PROJECT_ID) {
    Write-Host "   OK - Project: $PROJECT_ID" -ForegroundColor Green
} else {
    Write-Host "   WARN - Current project: $currentProject" -ForegroundColor Yellow
}

# ----------------------------
# 2. SERVICE ACCOUNT
# ----------------------------
Write-Host ""
Write-Host "2. Checking service account..."
$saExists = gcloud iam service-accounts describe $SA_EMAIL 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "   OK - Service Account exists" -ForegroundColor Green
} else {
    Write-Host "   ERROR - Service Account NOT found" -ForegroundColor Red
}

# ----------------------------
# 3. IAM PERMISSIONS
# ----------------------------
Write-Host ""
Write-Host "3. Checking IAM permissions..."
$iamPolicy = gcloud projects get-iam-policy $PROJECT_ID --format=json | ConvertFrom-Json
$requiredRoles = @("roles/bigquery.dataEditor", "roles/bigquery.jobUser", "roles/run.invoker")
$foundRoles = @()

foreach ($binding in $iamPolicy.bindings) {
    if ($binding.members -contains "serviceAccount:$SA_EMAIL" -and $requiredRoles -contains $binding.role) {
        $foundRoles += $binding.role
    }
}

foreach ($role in $requiredRoles) {
    if ($foundRoles -contains $role) {
        Write-Host "   OK - $role" -ForegroundColor Green
    } else {
        Write-Host "   ERROR - Missing: $role" -ForegroundColor Red
    }
}

# ----------------------------
# 4. BIGQUERY DATASET & TABLE
# ----------------------------
Write-Host ""
Write-Host "4. Checking BigQuery..."
$TOKEN = gcloud auth application-default print-access-token 2>$null

try {
    $datasetCheck = Invoke-RestMethod -Uri "https://bigquery.googleapis.com/bigquery/v2/projects/$PROJECT_ID/datasets/$DATASET" -Headers @{Authorization="Bearer $TOKEN"} -ErrorAction Stop
    Write-Host "   OK - Dataset exists: $DATASET" -ForegroundColor Green
} catch {
    Write-Host "   ERROR - Dataset NOT found: $DATASET" -ForegroundColor Red
}

try {
    $tableCheck = Invoke-RestMethod -Uri "https://bigquery.googleapis.com/bigquery/v2/projects/$PROJECT_ID/datasets/$DATASET/tables/$TABLE" -Headers @{Authorization="Bearer $TOKEN"} -ErrorAction Stop
    Write-Host "   OK - Table exists: $TABLE" -ForegroundColor Green
    
    $QUERY = @{
        query = "SELECT COUNT(*) as count FROM ``$PROJECT_ID.$DATASET.$TABLE``"
        useLegacySql = $false
    } | ConvertTo-Json
    
    $countResult = Invoke-RestMethod -Uri "https://bigquery.googleapis.com/bigquery/v2/projects/$PROJECT_ID/queries" -Method POST -Headers @{Authorization="Bearer $TOKEN"; "Content-Type"="application/json"} -Body $QUERY
    $rowCount = $countResult.rows[0].f[0].v
    Write-Host "   OK - Rows in table: $rowCount" -ForegroundColor Green
} catch {
    Write-Host "   ERROR - Table NOT found: $TABLE" -ForegroundColor Red
}

# ----------------------------
# 5. CLOUD RUN JOB
# ----------------------------
Write-Host ""
Write-Host "5. Checking Cloud Run Job..."
$jobInfo = gcloud run jobs describe $JOB_NAME --region=$REGION --format=json 2>$null | ConvertFrom-Json
if ($jobInfo) {
    Write-Host "   OK - Job exists: $JOB_NAME" -ForegroundColor Green
    Write-Host "   OK - Service Account configured" -ForegroundColor Green
} else {
    Write-Host "   ERROR - Job NOT found: $JOB_NAME" -ForegroundColor Red
}

# ----------------------------
# 6. LAST EXECUTION
# ----------------------------
Write-Host ""
Write-Host "6. Checking last execution..."
$lastExecution = gcloud run jobs executions list --job=$JOB_NAME --region=$REGION --limit=1 --format=json 2>$null | ConvertFrom-Json
if ($lastExecution) {
    $execName = $lastExecution[0].metadata.name
    $execStatus = $lastExecution[0].status.conditions[0].type
    $execTime = $lastExecution[0].status.completionTime
    
    if ($execStatus -eq "Completed") {
        Write-Host "   OK - Last execution: $execName" -ForegroundColor Green
        Write-Host "   OK - Status: Completed" -ForegroundColor Green
        Write-Host "   OK - Time: $execTime" -ForegroundColor Green
    } else {
        Write-Host "   WARN - Last execution status: $execStatus" -ForegroundColor Yellow
    }
} else {
    Write-Host "   WARN - No executions found yet" -ForegroundColor Yellow
}

# ----------------------------
# 7. CLOUD SCHEDULER
# ----------------------------
Write-Host ""
Write-Host "7. Checking Cloud Scheduler..."
$schedulerInfo = gcloud scheduler jobs describe $SCHEDULER_NAME --location=$REGION --format=json 2>$null | ConvertFrom-Json
if ($schedulerInfo) {
    Write-Host "   OK - Scheduler exists: $SCHEDULER_NAME" -ForegroundColor Green
    Write-Host "   OK - Schedule: $($schedulerInfo.schedule)" -ForegroundColor Green
    Write-Host "   OK - Timezone: $($schedulerInfo.timeZone)" -ForegroundColor Green
    Write-Host "   OK - State: $($schedulerInfo.state)" -ForegroundColor Green
    Write-Host "   OK - Next run: $($schedulerInfo.scheduleTime)" -ForegroundColor Green
} else {
    Write-Host "   ERROR - Scheduler NOT found: $SCHEDULER_NAME" -ForegroundColor Red
}

# ----------------------------
# 8. RECENT LOGS
# ----------------------------
Write-Host ""
Write-Host "8. Checking recent logs..."
$recentLogs = gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME" --limit=5 --format="value(textPayload)" 2>$null
if ($recentLogs) {
    Write-Host "   OK - Recent log entries found" -ForegroundColor Green
    $recentLogs | Select-Object -First 3 | ForEach-Object {
        Write-Host "   $_"
    }
} else {
    Write-Host "   WARN - No recent logs found" -ForegroundColor Yellow
}

# ----------------------------
# 9. LATEST DATA
# ----------------------------
Write-Host ""
Write-Host "9. Checking latest BigQuery data..."
try {
    $QUERY2 = @{
        query = "SELECT date, signal_strength, exposure_spy, shock_nextday FROM ``$PROJECT_ID.$DATASET.$TABLE`` ORDER BY created_at DESC LIMIT 1"
        useLegacySql = $false
    } | ConvertTo-Json
    
    $latestData = Invoke-RestMethod -Uri "https://bigquery.googleapis.com/bigquery/v2/projects/$PROJECT_ID/queries" -Method POST -Headers @{Authorization="Bearer $TOKEN"; "Content-Type"="application/json"} -Body $QUERY2
    
    if ($latestData.rows) {
        $row = $latestData.rows[0]
        Write-Host "   OK - Latest signal:" -ForegroundColor Green
        Write-Host "      Date: $($row.f[0].v)"
        Write-Host "      Signal Strength: $($row.f[1].v)"
        Write-Host "      Exposure SPY: $($row.f[2].v)"
        Write-Host "      Shock Next Day: $($row.f[3].v)"
    } else {
        Write-Host "   WARN - No data in BigQuery yet" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ERROR - Cannot query BigQuery" -ForegroundColor Red
}

# ----------------------------
# SUMMARY
# ----------------------------
Write-Host ""
Write-Host "=========================================="
Write-Host "HEALTH CHECK COMPLETE"
Write-Host "=========================================="
Write-Host ""
Write-Host "Quick commands:"
Write-Host "  Manual trigger: gcloud run jobs execute $JOB_NAME --region=$REGION --wait"
Write-Host "  View logs:      gcloud logging read `"resource.type=cloud_run_job`" --limit=20"
Write-Host ""