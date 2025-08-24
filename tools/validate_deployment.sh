#!/bin/bash

# ML-6 Deployment Validation Script
# Usage: ./tools/validate_deployment.sh [SERVICE_URL] [ENVIRONMENT]

SERVICE_URL=${1:-"https://solaroptima-ml-dev-xxxxx-ew.a.run.app"}
ENVIRONMENT=${2:-"dev"}

echo "🔍 Validating ML-6 deployment for $ENVIRONMENT environment"
echo "Service URL: $SERVICE_URL"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local expected_status=$2
    local description=$3
    
    echo -n "Testing $description... "
    
    response=$(curl -s -o /tmp/response.json -w "%{http_code}" "$SERVICE_URL$endpoint" 2>/dev/null)
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        if [ "$endpoint" = "/health" ]; then
            echo "   Response: $(cat /tmp/response.json)"
        fi
    else
        echo -e "${RED}❌ FAIL (got $response, expected $expected_status)${NC}"
        echo "   Response: $(cat /tmp/response.json 2>/dev/null || echo 'No response body')"
    fi
}

# Function to test model3d endpoint with sample data
test_model3d() {
    echo -n "Testing /model3d endpoint... "
    
    # Create a minimal test request
    cat > /tmp/test_request.json << EOF
{
    "bbox": [51.5074, -0.1278, 51.5075, -0.1277],
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
}
EOF
    
    response=$(curl -s -o /tmp/model3d_response.json -w "%{http_code}" \
        -X POST "$SERVICE_URL/model3d" \
        -H "Content-Type: application/json" \
        -d @/tmp/test_request.json 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        echo "   Response keys: $(jq -r 'keys | join(", ")' /tmp/model3d_response.json 2>/dev/null || echo 'No JSON response')"
    else
        echo -e "${RED}❌ FAIL (got $response, expected 200)${NC}"
        echo "   Response: $(cat /tmp/model3d_response.json 2>/dev/null || echo 'No response body')"
    fi
}

# Function to check environment variables
check_env_vars() {
    echo -n "Checking environment variables... "
    
    # Get service info
    service_info=$(curl -s "$SERVICE_URL/" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service responding${NC}"
        echo "   Available endpoints: $(echo $service_info | jq -r '.endpoints | join(", ")' 2>/dev/null || echo 'Unknown')"
    else
        echo -e "${RED}❌ Service not responding${NC}"
    fi
}

# Function to check Cloud Run service status
check_cloud_run_status() {
    echo -n "Checking Cloud Run service status... "
    
    # Extract service name from URL
    service_name=$(echo $SERVICE_URL | sed 's/.*\/\([^\/]*\)-[a-z0-9]*-[a-z0-9]*\.a\.run\.app.*/\1/')
    region=$(echo $SERVICE_URL | sed 's/.*-\([a-z0-9]*\)\.a\.run\.app.*/\1/')
    
    if [ -n "$service_name" ] && [ -n "$region" ]; then
        status=$(gcloud run services describe $service_name --region=$region --format="value(status.conditions[0].status)" 2>/dev/null)
        
        if [ "$status" = "True" ]; then
            echo -e "${GREEN}✅ Service is ready${NC}"
        else
            echo -e "${YELLOW}⚠️  Service status: $status${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Could not extract service info from URL${NC}"
    fi
}

# Main validation
echo "=== Basic Health Checks ==="
check_endpoint "/health" "200" "Health endpoint"
check_endpoint "/" "200" "Root endpoint"
check_endpoint "/docs" "200" "API documentation"

echo ""
echo "=== Service Information ==="
check_env_vars

echo ""
echo "=== Cloud Run Status ==="
check_cloud_run_status

echo ""
echo "=== Model Endpoint Tests ==="
test_model3d

echo ""
echo "=== Artifact Storage Test ==="
echo -n "Checking artifact storage... "
# This would require actual model3d request to test artifact generation
echo -e "${YELLOW}⚠️  Manual verification needed${NC}"
echo "   Run a /model3d request and check GCS bucket: gs://ml6-artifacts-$ENVIRONMENT"

echo ""
echo "=== Performance Check ==="
echo -n "Testing response time... "
start_time=$(date +%s.%N)
curl -s "$SERVICE_URL/health" > /dev/null
end_time=$(date +%s.%N)
response_time=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "unknown")
echo -e "${GREEN}✅ ${response_time}s${NC}"

echo ""
echo "=== Summary ==="
echo "✅ Basic validation complete"
echo ""
echo "Next steps:"
echo "1. Upload ML models to gs://ml6-models-$ENVIRONMENT"
echo "2. Test with real aerial imagery via /model3d"
echo "3. Verify artifacts are generated in gs://ml6-artifacts-$ENVIRONMENT"
echo "4. Set up monitoring and alerting"
echo "5. Configure CORS for your frontend domains"

# Cleanup
rm -f /tmp/response.json /tmp/model3d_response.json /tmp/test_request.json 