import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from src import config

class TextEncoder(nn.Module):
    """
    Text encoder using pretrained CLIP text model.
    Converts a list of strings into fixed-size embeddings.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        # Load tokenizer and model
        self.tokenizer = CLIPTokenizer.from_pretrained(config.CLIP_MODEL_NAME)
        self.model = CLIPTextModel.from_pretrained(config.CLIP_MODEL_NAME)

        # Move model to the specified device
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: list) -> torch.Tensor:
        """
        Encode a batch of text strings into embeddings.

        Args:
            texts: List of text strings.

        Returns:
            Tensor of shape (batch_size, config.TEXT_EMBEDDING_DIM)
        """
        # Tokenize input texts
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Get token embeddings and average over sequence length
        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings
