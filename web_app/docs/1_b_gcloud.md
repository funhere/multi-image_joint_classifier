```

GCP_PROJECT=....
APP_NAME=usuke_classification
REGION="us-central1"
MEMORY=2G
gcloud builds submit --tag gcr.io/$GCP_PROJECT/$APP_NAME


gcloud beta run deploy $APP_NAME --image gcr.io/$GCP_PROJECT/$APP_NAME --region $REGION --memory $MEMORY --allow-unauthenticated

```
