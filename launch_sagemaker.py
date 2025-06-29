import sagemaker
from sagemaker.huggingface import HuggingFace

# Set your S3 paths
input_s3_uri = 's3://souritra-s-bucket/Sentiment-Analysis/input/Amazon_Product_Reviews_US.csv'
output_s3_uri = 's3://souritra-s-bucket/Sentiment-Analysis/output/'

# Role ARN from previous step
role = 'arn:aws:iam::790856971345:role/sagemaker-execution-role'

# HuggingFace estimator setup
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.g4dn.xlarge",           # CPU
    instance_count=1,
    transformers_version="4.6.1",          # <--- KEY! Supported for CPU
    pytorch_version="1.7.1",               # <--- KEY! Supported for CPU
    py_version="py36",                     # <--- KEY! Supported for CPU
    role=role,
    hyperparameters = {'epochs': 2, 'learning_rate': 0.001}
)



# Launch the training job
huggingface_estimator.fit({'train': input_s3_uri})
