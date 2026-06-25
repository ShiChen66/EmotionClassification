# push model to huggingface hub

from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path = './model_output',
    repo_id = 'ShiChenLee/EmotionClassification',
    repo_type = 'model'
)

print('Model pushed successfully.')