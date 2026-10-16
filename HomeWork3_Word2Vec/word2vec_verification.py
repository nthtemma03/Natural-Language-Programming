from multiprocessing import context

import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2VecSGNS(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super(Word2VecSGNS, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Target vectors (W) and Context vectors (W_prime)
        self.W_target = nn.Embedding(vocab_size, embed_dim)
        self.W_context = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize vectors uniformly or normally
        initrange = 1.0 / embed_dim
        self.W_target.weight.data.uniform_(-initrange, initrange)
        self.W_context.weight.data.uniform_(-initrange, initrange)

    def forward(self, target, context, negatives):
        """
        Computes the SGNS loss using standard PyTorch operations.
        
        Args:
            target:    Tensor of shape (batch_size,)
            context:   Tensor of shape (batch_size,)
            negatives: Tensor of shape (batch_size, num_negatives)
        Returns:
            loss: Scalar tensor representing the batch mean SGNS loss
        """
        target_vec = self.W_target( target) # (batch_size, embed_dim)
        context_vec = self.W_context(context) # (batch_size, embed_dim)
        negative_vec = self.W_context(negatives) # (batch_size, num_negatives, embed_dim)

        # Positive score: dot product of target and context
        pos_score = torch.sum(target_vec * context_vec, dim=1) 
        pos_loss = F.logsigmoid(pos_score)  # log(sigmoid(pos_score))

        # Negative score: dot product of target and negative samples
        neg_score = torch.bmm(negative_vec, target_vec.unsqueeze(2)).squeeze(2)

        neg_loss = F.logsigmoid(-neg_score).sum(1)  # log(sigmoid(-neg_score))
        loss = -(pos_loss + neg_loss).mean()  # Mean over the batch
        return loss
        print("FORWARD CALLED")

    def manual_backward(self, target, context, negatives):
        """
        COMPUTE BACKPROPAGATION BY HAND.
        
        Calculate gradients analytically using PyTorch Tensors, 
        but strictly WITHOUT invoking autograd (.backward()).
        
        Returns:
            grad_W_target: Tensor of shape (vocab_size, embed_dim) filled with 
                           computed manual gradients for the target matrix.
            grad_W_context: Tensor of shape (vocab_size, embed_dim) filled with
                            computed manual gradients for the context matrix.
        """
        grad_W_target = torch.zeros_like(self.W_target.weight.data)
        grad_W_context = torch.zeros_like(self.W_context.weight.data)
        
        with torch.no_grad():
            # TODO: Extract necessary embedding vectors
            # TODO: Compute sigmoids and analytical probability errors
            # TODO: Accumulate gradients back to their respective row indices
            # Hint: Take account of the batch average scaling factor (1 / batch_size)!
            batch_size = target.size(0)

            target_vec = self.W_target.weight[target]
            context_vec = self.W_context.weight[context]
            negative_vec = self.W_context.weight[negatives]

            # Positive part
            pos_score = torch.sum(
                target_vec * context_vec,dim=1)

            pos_sigmoid = torch.sigmoid(pos_score)

            g_pos = (pos_sigmoid - 1.0) / batch_size

            # Negative part
            neg_score = torch.sum(
            negative_vec * target_vec.unsqueeze(1),dim=2)

            neg_sigmoid = torch.sigmoid(neg_score)

            g_neg = neg_sigmoid / batch_size

            for b in range(batch_size):

                t_idx = target[b].item()
                c_idx = context[b].item()

                # positive contribution

                grad_W_target[t_idx] += (
                    g_pos[b] * context_vec[b]\
                )

                grad_W_context[c_idx] += (
                    g_pos[b] * target_vec[b]
                )

                # negative contributions

                for k in range(negatives.size(1)):

                    n_idx = negatives[b, k].item()

                    grad_W_target[t_idx] += (
                        g_neg[b, k] * negative_vec[b, k]
                    )

                    grad_W_context[n_idx] += (
                        g_neg[b, k] * target_vec[b]
                    )
        return grad_W_target, grad_W_context

def verify_gradients():
    """
    Unit test rig acting as the gradient auditor suite. Do not modify.
    """
    print("--- Starting Analytical Gradient Verification Test ---")
    vocab_size = 50
    embed_dim = 10
    num_negatives = 3
    
    model = Word2VecSGNS(vocab_size, embed_dim)
    
    # Mock data batch (batch_size = 4)
    target = torch.tensor([1, 5, 12, 44], dtype=torch.long)
    context = torch.tensor([2, 8, 15, 3], dtype=torch.long)
    negatives = torch.tensor([
        [10, 20, 30],
        [4, 14, 24],
        [7, 17, 27],
        [9, 19, 29]
    ], dtype=torch.long)
    
    # --- Step 1: Framework Autograd Pass ---
    model.zero_grad()
    pytorch_loss = model(target, context, negatives)
    pytorch_loss.backward()
    
    auto_grad_target = model.W_target.weight.grad.clone()
    auto_grad_context = model.W_context.weight.grad.clone()
    
    # --- Step 2: Student Manual Pass ---
    manual_grad_target, manual_grad_context = model.manual_backward(target, context, negatives)
    
    # --- Step 3: Verification Checks ---
    target_match = torch.allclose(auto_grad_target, manual_grad_target, atol=1e-5)
    context_match = torch.allclose(auto_grad_context, manual_grad_context, atol=1e-5)
    
    if target_match and context_match:
        print("✅ SUCCESS: Your manual backprop implementation matches PyTorch exactly!")
    else:
        if not target_match:
            print("❌ FAILURE: Gradients for W_target do not match.")
            print(f"Max absolute difference Target: {torch.max(torch.abs(auto_grad_target - manual_grad_target))}")
        if not context_match:
            print("❌ FAILURE: Gradients for W_context do not match.")
            print(f"Max absolute difference Context: {torch.max(torch.abs(auto_grad_context - manual_grad_context))}")
            
if __name__ == "__main__":
    verify_gradients()
