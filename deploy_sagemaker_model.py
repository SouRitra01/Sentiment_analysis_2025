# deploy_sagemaker_model.py

from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role, Session

# 1. S3 path of the trained model
model_data = "s3://sagemaker-ap-south-1-790856971345/huggingface-pytorch-training-2025-06-29-11-22-43-165/output/model.tar.gz"

# 2. IAM role used for SageMaker
role = "arn:aws:iam::790856971345:role/sagemaker-execution-role"

# 3. HuggingFace Model (match transformers & pytorch version from your training job)
huggingface_model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    transformers_version="4.26.0",    # or what you used in training
    pytorch_version="1.13.1",         # or what you used in training
    py_version="py39"                 # or "py310" if that's what you trained with
)

# 4. Deploy the model as an endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"       # use your allowed instance type, e.g., "ml.m5.large"
)

# 5. Run a test prediction
data = {
    "inputs": "This product is amazing! Highly recommended."
}
result = predictor.predict(data)
print("Prediction result:", result)

# 6. Clean up (delete the endpoint when done to avoid costs!)
# predictor.delete_endpoint()
