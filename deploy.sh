#!/usr/bin/env bash
# ============================================================
#  I.V.S.S. — Google Cloud Run Deployment Script
#  Usage: bash deploy.sh
# ============================================================
set -euo pipefail

# ── Configuration (edit these) ───────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="asia-south1"                    # Mumbai (closest to India)
SERVICE_NAME="ivss"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# ── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ── Pre-flight checks ───────────────────────────────────────
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud CLI not found.${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}GCP_PROJECT_ID not set. Attempting to detect...${NC}"
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}ERROR: No GCP project configured.${NC}"
        echo "Run: gcloud config set project YOUR_PROJECT_ID"
        echo "Or:  export GCP_PROJECT_ID=YOUR_PROJECT_ID"
        exit 1
    fi
    IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
    echo -e "${GREEN}Using project: ${PROJECT_ID}${NC}"
fi

# ── Step 1: Set project & enable APIs ────────────────────────
echo -e "\n${GREEN}[1/4] Setting project & enabling APIs...${NC}"
gcloud config set project "$PROJECT_ID"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com

# ── Step 2: Build Docker image with Cloud Build ──────────────
echo -e "\n${GREEN}[2/4] Building Docker image with Cloud Build...${NC}"
echo -e "${YELLOW}(This may take 5-10 min on first build due to PyTorch)${NC}"
gcloud builds submit --tag "$IMAGE_NAME" --timeout=1800

# ── Step 3: Read secrets from .env ───────────────────────────
echo -e "\n${GREEN}[3/4] Reading environment variables from .env...${NC}"
ENV_VARS="FLASK_ENV=production"
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
        # Extract KEY=VALUE
        key=$(echo "$line" | cut -d'=' -f1 | xargs)
        value=$(echo "$line" | cut -d'=' -f2- | xargs)
        if [ -n "$key" ] && [ -n "$value" ]; then
            ENV_VARS="${ENV_VARS},${key}=${value}"
        fi
    done < .env
    echo -e "${GREEN}Loaded secrets from .env${NC}"
else
    echo -e "${YELLOW}WARNING: No .env file found. Using defaults.${NC}"
fi

# ── Step 4: Deploy to Cloud Run ──────────────────────────────
echo -e "\n${GREEN}[4/4] Deploying to Cloud Run...${NC}"
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_NAME" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 1 \
    --timeout 300 \
    --session-affinity \
    --set-env-vars "$ENV_VARS" \
    --port 8080

# ── Done! ────────────────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="value(status.url)")
echo ""
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ I.V.S.S. deployed successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "  URL:    ${YELLOW}${SERVICE_URL}${NC}"
echo -e "  Health: ${YELLOW}${SERVICE_URL}/health${NC}"
echo -e "  Logs:   gcloud run logs read --service ${SERVICE_NAME} --region ${REGION}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
