import torch


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Triplet loss using cosine similarity.
        Loss = max(0, sim(anchor, neg) - sim(anchor, pos) + margin)
        Pushes negatives away and positives close.
        
        Args:
            anchor: (batch_size, embedding_dim)
            positive: (batch_size, embedding_dim)
            negative: (batch_size, embedding_dim)
        
        Returns:
            scalar loss
        """
        cosine_positive = torch.nn.functional.cosine_similarity(anchor, positive, dim=-1)
        cosine_negative = torch.nn.functional.cosine_similarity(anchor, negative, dim=-1)
        
        # Loss: push negative away, positive close
        losses = torch.relu(cosine_negative - cosine_positive + self.margin)
        return losses.mean()
