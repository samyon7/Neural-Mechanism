import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SlidingWindowAttention(nn.Module): #SWA
  #Implement Sliding Window Attention
    def __init__(self, dim, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 1. Compute query, key, and value in one forward pass
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Initialize the final output
        output = torch.zeros_like(x)

        # Loop through the sequence within the window boundaries
        for i in range(seq_len):
          # Determine start and end of the window
          start = max(0, i-self.window_size//2) # Integer division to get an integer
          end = min(seq_len, i+self.window_size//2 + 1) # Increment the iterator

          # 2. Compute the window
          window_keys = keys[:, start:end, :]
          window_values = values[:, start:end, :]
          window_query = queries[:, i:i+1, :]

          # 3. Compute dot product attention
          attention_scores = torch.matmul(window_query, window_keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32)) # Normalization
          attention_weights = F.softmax(attention_scores, dim=-1)

          # 4. Compute weighted values and sum them
          weighted_values = torch.matmul(attention_weights, window_values)

          # Update output vector
          output[:, i, :] = weighted_values.squeeze(1)

        # Return the result
        return output

class LongTermMemory(nn.Module):
    def __init__(self, dim, key_value_dim, num_persistent_weights, learning_rate=0.001,
                 past_surprise_weight=0.9, forget_rate=0.001, window_size=4, chunk_size=4):

        super().__init__()
        self.dim = dim
        self.key_value_dim = key_value_dim
        self.learning_rate = learning_rate
        self.past_surprise_weight = past_surprise_weight
        self.forget_rate = forget_rate
        self.window_size = window_size # Add sliding window size
        self.chunk_size = chunk_size

        # Linear projections for key and value
        self.key_proj = nn.Linear(dim, key_value_dim)
        self.value_proj = nn.Linear(dim, key_value_dim)
        self.query_proj = nn.Linear(dim, key_value_dim) # Projection for memory retrieval

        # Persistent Memory (moved to initialization)
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent_weights, dim)) # Input dimension

        # Neural Memory as a Layer and Gated Mechanism
        self.neural_memory = nn.Sequential(
            nn.Linear(dim, key_value_dim),
            nn.ReLU(),
            nn.Linear(key_value_dim, dim)
        )
        # Forget Gate
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # Sliding Window Attention
        self.swa = SlidingWindowAttention(dim, window_size)

        # Normalization (Learnable vector-valued weight)
        self.norm_weight = nn.Parameter(torch.ones(dim))

        # Gating Mechanism
        self.gate = nn.Linear(2*dim, 1) # Combining SWA output and Neural Memory

        # Loss function:
        self.loss_fn = nn.MSELoss() # Loss function (MSELoss)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input_sequence): # Process

        # 1. Reshape input with a new batch (for parallel processing, into Linear Within-Chunk)
        batch_size, seq_len, _ = input_sequence.shape
        persistent_len = self.persistent_memory.shape[0]
        input_sequence_in_batch = input_sequence

        # 2. Projection
        key = self.key_proj(input_sequence_in_batch)
        value = self.value_proj(input_sequence_in_batch)

        # 3. Compute Loss (Surprise) and Gradient (approximate momentary surprise)
        loss = self.loss_fn(key, value)
        loss.backward(retain_graph=True)
        momentary_surprise = self.key_proj.weight.grad.clone()
        self.key_proj.weight.grad.zero_()

        # 4. Compute Past Surprise
        if not hasattr(self, 'past_surprise'):
            self.past_surprise = torch.zeros_like(momentary_surprise)
        self.past_surprise = (self.past_surprise_weight * self.past_surprise) - (self.learning_rate * momentary_surprise) # Update

        # 5. Forget Gate
        with torch.no_grad():
          self.alpha.data = torch.clamp(self.alpha.data + self.forget_rate, 0, 1)
          self.key_proj.weight.data = (1 - self.alpha) * self.key_proj.weight + self.past_surprise # Equation 13 Update

        # 6. Sliding Window Attention
        M_data = self.neural_memory(input_sequence_in_batch)
        o = self.swa(M_data) # SW Attention

        return o

if __name__ == '__main__':
    dim = 64
    key_value_dim = 32
    num_persistent_weights = 4
    learning_rate = 0.001
    window_size = 4

    # Initialize model
    memory_as_layer = LongTermMemory(dim, key_value_dim, num_persistent_weights, learning_rate, window_size=4)

    # Generate input
    batch_size = 2
    seq_len = 16
    input_sequence = torch.randn(batch_size, seq_len, dim)

    # Train with Adam optimizer
    optimizer = optim.Adam(memory_as_layer.parameters())

    # Train
    o = memory_as_layer(input_sequence)
    print(o.shape)
