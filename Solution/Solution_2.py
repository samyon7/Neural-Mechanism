"""
This code's a long-term memory update rig predicated on surprise. 
The memory is calculated from the divergence between memory and fresh data. 
Memory morphs incrementally to accommodate incoming data over time.
"""


import numpy as np

class LongTermMemory:
    def __init__(self, dim, learning_rate=0.01, past_surprise_weight=0.9):
        self.dim = dim
        self.learning_rate = learning_rate
        self.past_surprise_weight = past_surprise_weight
        self.memory = np.zeros(dim)  # Initialize memory
        self.past_surprise = np.zeros(dim)  # Initialize past surprise
    
    # MSE
    def mse_loss(self, memory, input_data):
        return np.mean((memory - input_data) ** 2)
    
    # Derivation of MSE facing memory
    def gradient(self, memory, input_data):
        return 2 * (memory - input_data) / len(memory)
    
    def forward(self, input_data):
        # 1. Compute Loss
        loss = self.mse_loss(self.memory, input_data)
        
        # 2. Compute gradient (approximation of momentary surprise)
        momentary_surprise = self.gradient(self.memory, input_data)
        
        # 3. Update Past Surprise
        self.past_surprise = (self.past_surprise_weight * self.past_surprise) - (self.learning_rate * momentary_surprise)
        
        # 4. Update Memory
        self.memory += self.past_surprise
        
        return self.memory, loss

if __name__ == '__main__':
    dim = 10  # Memory dimension
    learning_rate = 0.1
    past_surprise_weight = 0.9

    # Initialize Memory Module
    ltm = LongTermMemory(dim, learning_rate, past_surprise_weight)

    # Generate random input data
    num_steps = 5000
    input_data = [np.random.randn(dim) for _ in range(num_steps)]

    # Process data
    for i, data in enumerate(input_data):
        memory, loss = ltm.forward(data)
        print(f"Step {i+1}: Loss = {loss:.4f}, Memory = {np.linalg.norm(memory):.4f}")
