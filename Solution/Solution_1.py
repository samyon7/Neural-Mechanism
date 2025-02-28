import numpy as np

class LinearAttentionWithForgetGate:
    def __init__(self, dim):
        self.dim = dim
        self.forget_gate_weights = np.random.randn(dim, 1)  # Forget gate weights

    def forget_gate(self, keys):
        """Computes the forget gate using sigmoid."""
        return 1 / (1 + np.exp(-np.dot(keys, self.forget_gate_weights)))  # Sigmoid

    def forward(self, queries, keys, values):
        """Performs Linear Attention with Forget Gate using numpy."""
        # 1. Kernel Transformation (using ReLU)
        Q = np.maximum(0, queries)  # ReLU
        K = np.maximum(0, keys)  # ReLU
        
        # 2. Forget Gate (determines how much memory to forget)
        forget_weights = self.forget_gate(K)  # (B, N, 1)
        
        # 3. Compute accumulator matrix with Forget Gate
        K_modified = K * forget_weights
        values_modified = values * forget_weights
        kv = np.einsum('bnd,bne->bde', K_modified, values_modified)
        
        # 4. Compute output: y_t = Q_t M_t
        output = np.einsum('bnd,bde->bne', Q, kv)
        
        # 5. Normalization
        Z = 1 / (np.einsum('bnd,bd->bn', Q, np.sum(K, axis=1)) + 1e-6)  # (batch_size, num_queries)
        Z = Z[:, :, np.newaxis]  # Reshape to (batch_size, num_queries, 1)

        output = output * Z
        
        return output
