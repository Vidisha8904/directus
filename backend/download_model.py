from transformers import AutoModel, AutoConfig
import traceback

try:
    model_name = "nomic-ai/nomic-embed-text-v1"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model {model_name} downloaded successfully to local cache.")
except Exception as e:
    print(f"Error downloading model: {str(e)}")
    traceback.print_exc()