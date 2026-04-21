# create a sample chat example using the llama model
from llama_cpp import Llama
import os
from dotenv import load_dotenv
load_dotenv()

model = Llama(model_path=os.getenv("MODEL_PATH"), n_gpu_layers=35, n_ctx=4096, verbose=False)

SYSTEM_PROMPT = """
You are an expert in finance, you are about to be asked a question by a user that may not have as much knowledge about financial concepts.
You will try to reword their question so the answer can be more easily searched for
"""
response = model.create_chat_completion(messages=[{"role": "user", "content": "What is the capital of France?"}])
print(response["choices"][0]["message"]["content"])