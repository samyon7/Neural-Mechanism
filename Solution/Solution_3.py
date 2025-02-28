"""
Losses are mooning, memory norms are bloatin' like crazy. 
Past-surprise is the perp – no regulation, keeps pumpin' up. Alpha's a joke; memory ain't forgettin' nothin', just keeps accumulatin'. 
No normalization = uncontrolled growth. Mess.

We also use the same setup, small-scale pretraining and fine-tuning. Identical algorithms. Loss exploding? Mitigate thusly.

Want to customize it? Use your eye to look the content below :
Reduce the learning rate → For example, lower the learning_rate to 0.001 so the updates are less aggressive.
Increase the forget rate (forget_rate) → For example, set forget_rate = 0.05 so the memory forgets old values faster.
Use normalization on the memory → After the memory is updated, clipping or another normalization technique can be applied.
"""



import numpy as np

class LongTermMemory:
    def __init__(self, dim, key_value_dim, learning_rate=0.01, past_surprise_weight=0.9, forget_rate=0.01):
        self.dim = dim  # Input dimension
        self.key_value_dim = key_value_dim  # Key & value dimension
        self.learning_rate = learning_rate
        self.past_surprise_weight = past_surprise_weight
        self.forget_rate = forget_rate

        # Linear projections for key and value (random weights)
        self.key_proj = np.random.randn(key_value_dim, dim)
        self.value_proj = np.random.randn(key_value_dim, dim)
        self.query_proj = np.random.randn(key_value_dim, dim)
        
        # Memory matrix
        self.memory = np.random.randn(key_value_dim, key_value_dim)

        # Forget Gate
        self.alpha = 0.0  # Initialize α with 0
        
        # Past surprise initialization
        self.past_surprise = np.zeros((key_value_dim, key_value_dim))
    
    def forward(self, input_data):
        """
        Processes input and updates memory.
        """
        # 1. Project to Key and Value
        key = np.dot(self.key_proj, input_data)
        value = np.dot(self.value_proj, input_data)

        # 2. Predict Value from Memory
        predicted_value = np.dot(self.memory, key)  # M(k_t)

        # 3. Compute Loss (MSE)
        loss = np.mean((predicted_value - value) ** 2)

        # 4. Compute Gradient (approximate momentary surprise)
        momentary_surprise = np.outer(predicted_value - value, key)

        # 5. Update Past Surprise
        self.past_surprise = (self.past_surprise_weight * self.past_surprise) - (self.learning_rate * momentary_surprise)

        # 6. Update Memory (with forget gate)
        self.alpha = np.clip(self.alpha + self.forget_rate, 0, 1)  # alpha -> 1 means more forgetting
        self.memory = (1 - self.alpha) * self.memory + self.past_surprise  # Equation 13

        return self.memory, loss
    
    def retrieve(self, input_data):
        """
        Retrieve memory corresponding to the input (query).
        """
        query = np.dot(self.query_proj, input_data)
        retrieved_memory = np.dot(self.memory, query)  # yt = M * qt
        return retrieved_memory

# Example Usage
if __name__ == '__main__':
    dim = 64  # Input dimension
    key_value_dim = 32  # Key and Value dimension
    learning_rate = 0.01
    past_surprise_weight = 0.9
    forget_rate = 0.001

    # Initialize Memory Module
    ltm = LongTermMemory(dim, key_value_dim, learning_rate, past_surprise_weight, forget_rate)

    # Generate random input data
    num_steps = 1000
    input_data = [np.random.randn(dim) for _ in range(num_steps)]

    # Process data and train
    for i, data in enumerate(input_data):
        memory, loss = ltm.forward(data)
        print(f"Step {i+1}: Loss = {loss:.4f}, Memory = {np.linalg.norm(memory):.4f}, Forget = {ltm.alpha:.4f}")

    # Retrieve memory (inference):
    test_data = np.random.randn(dim)
    retrieved_memory = ltm.retrieve(test_data)
    print("Retrieved Memory Shape:", retrieved_memory.shape)  # Verify output shape
