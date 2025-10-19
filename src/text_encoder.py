import torch
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)
        return embeddings  # shape: (batch, 512)
