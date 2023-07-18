from rich.console import Console
from rich.markdown import Markdown

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "You are a helpful AI assistant", history=[])
console = Console()

while True:
    text = input("👨: ")
    response, history = model.chat(tokenizer, text, history=history)
    console.print(Markdown(response))