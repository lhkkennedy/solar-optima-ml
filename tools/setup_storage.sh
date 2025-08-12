#!/bin/bash

# ML-6 Storage Setup Script
# Usage: ./tools/setup_storage.sh [PROJECT_ID] [REGION]

PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"europe-west2"}

echo "Setting up ML-6 storage for project: $PROJECT_ID in region: $REGION"

# Create buckets for each environment
for env in dev staging; do
    echo "Creating buckets for $env environment..."
    
    # Models bucket
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://ml6-models-$env
    
    # Artifacts bucket
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://ml6-artifacts-$env
    
    # Set lifecycle policy for artifacts (optional: delete after 30 days)
    cat > /tmp/lifecycle.json << EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 30,
        "matchesPrefix": ["artifacts/"]
      }
    }
  ]
}
EOF
    gsutil lifecycle set /tmp/lifecycle.json gs://ml6-artifacts-$env
    
    echo "✅ Created buckets for $env:"
    echo "   - gs://ml6-models-$env"
    echo "   - gs://ml6-artifacts-$env"
done

# Set up IAM permissions (adjust service account names as needed)
echo "Setting up IAM permissions..."

# For development environment
gsutil iam ch serviceAccount:ml6-deploy-dev@$PROJECT_ID.iam.gserviceaccount.com:objectViewer gs://ml6-models-dev
gsutil iam ch serviceAccount:ml6-deploy-dev@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://ml6-artifacts-dev

# For staging environment
gsutil iam ch serviceAccount:ml6-deploy-staging@$PROJECT_ID.iam.gserviceaccount.com:objectViewer gs://ml6-models-staging
gsutil iam ch serviceAccount:ml6-deploy-staging@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://ml6-artifacts-staging

echo "✅ Storage setup complete!"
echo ""
echo "Next steps:"
echo "1. Update GitHub secrets with bucket names:"
echo "   - GCS_MODELS_BUCKET_DEV=ml6-models-dev"
echo "   - GCS_ARTIFACTS_BUCKET_DEV=ml6-artifacts-dev"
echo "   - GCS_MODELS_BUCKET_STAGING=ml6-models-staging"
echo "   - GCS_ARTIFACTS_BUCKET_STAGING=ml6-artifacts-staging"
echo "2. Upload your ML models to the models buckets"
echo "3. Configure GitHub environments and secrets" 