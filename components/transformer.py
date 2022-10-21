import torch
from torch import nn
from torch.nn import functional


class Transformer(nn.Module):
    """
    Transformer architecture
    """

    def __init__(self, n_channel: int, n_hidden: int) -> None:
        super().__init__()
        self.f_1 = nn.Linear(n_channel, n_channel, bias=False)
        self.f_2 = nn.Linear(n_channel, n_channel, bias=False)
        self.key = nn.Linear(n_channel, n_hidden)
        self.query = nn.Linear(n_channel, n_hidden)

    def forward(self, visual_token: torch.Tensor) -> torch.Tensor:
        """The transformer is a self attention module, modelling the
        relationship between the different visual token

        Args:
            visual_token (torch.Tensor): The visual tokens calculated by the
            tokenizer, has the shape (batch_size, n_token, n_channel)

        Returns:
            torch.Tensor: Visual tokens after applied self attention, has the
            shape (batch_size, n_token, n_channel)
        """
        visual_token_k = self.key(visual_token)
        visual_token_q = self.query(visual_token)
        self_attention = functional.softmax(
            torch.matmul(visual_token_k, visual_token_q.transpose(1, 2)), dim=1
        )
        visual_token_p = visual_token + torch.matmul(self_attention, visual_token)
        visual_token_self_attention = visual_token_p + functional.relu(
            self.f_2(self.f_1(visual_token_p))
        )
        return visual_token_self_attention


if __name__ == "__main__":
    transformer = Transformer(256, 32)
    for p in transformer.parameters():
        print(p.shape)
