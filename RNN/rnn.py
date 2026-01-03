import numpy as np
np.random.seed(42)


class RNN:
    
    def __init__(self, input=1, hidden=100, seq_len=50):
        self.input = input
        self.hidden = hidden
        self.output = 1
        self.seq_len = seq_len
        
        self.w_xh = np.random.randn(self.input, self.hidden) * 0.1      # (D, H)
        self.w_hh = np.random.randn(self.hidden, self.hidden) * 0.1     # (H, H)
        self.w_hy = np.random.randn(self.hidden, self.output) * 0.1     # (H, 1)
        
        self.b_h = np.random.rand(self.hidden)
        self.b_y = np.random.rand(self.input)
        
        
    def forward(self, x):
        self.x = x
        self.h_t = np.zeros((self.seq_len + 1, self.hidden))  # h_t[0] = 0
        self.a_t = np.zeros((self.seq_len, self.hidden))
        
        for t in range(1, self.seq_len + 1):
            self.a_t[t-1] = (self.x[t-1] @ self.w_xh) + (self.h_t[t-1] @ self.w_hh) + self.b_h
            self.h_t[t] = np.tanh(self.a_t[t-1])
        
        self.y_hat = (self.h_t[self.seq_len] @ self.w_hy + self.b_y).item()
        return self.y_hat
    
    def backward(self, e, learning_rate=0.001):
        dl_dw_hy = self.h_t[self.seq_len][:, None] * e     # (H,1)
        dl_db_y  = np.array([e])                           # (1,)
        
        dl_dh_t = (self.w_hy[:, 0] * e)                    # (H,)
        
        dl_dw_xh = np.zeros_like(self.w_xh)                # (D,H)
        dl_dw_hh = np.zeros_like(self.w_hh)                # (H,H)
        dl_db_h  = np.zeros_like(self.b_h)                 # (H,)
        
        for t in range(self.seq_len, 0, -1):
            delta_t = dl_dh_t * (1.0 - self.h_t[t] ** 2)

            dl_dw_xh += np.outer(self.x[t-1], delta_t)     # (D,H)
            dl_dw_hh += np.outer(self.h_t[t-1], delta_t)   # (H,H)
            dl_db_h  += delta_t                             # (H,)
            
            dl_dh_t = delta_t @ self.w_hh.T                # (H,)
            
        clip_norm = 5.0
        def clip(g):
            n = np.linalg.norm(g)
            if n > clip_norm:
                return g * (clip_norm / (n + 1e-12))
            return g
        
        dl_dw_xh = clip(dl_dw_xh)
        dl_dw_hh = clip(dl_dw_hh)
        dl_dw_hy = clip(dl_dw_hy)
        dl_db_h  = clip(dl_db_h)
        dl_db_y  = clip(dl_db_y)
        
        self.w_xh -= (learning_rate * dl_dw_xh)
        self.w_hh -= (learning_rate * dl_dw_hh)
        self.w_hy -= (learning_rate * dl_dw_hy)
        
        self.b_h -= (learning_rate * dl_db_h)
        self.b_y -= (learning_rate * dl_db_y)
            
rnn = RNN(input=1, hidden=100, seq_len=50)

scale = 1000.0
train_max_start = 900 

epochs = 20000
learning_rate = 5e-3

inp = (np.arange(1, 51) / 50).reshape(50, 1)
tar = 51/50

epochs = 20000

for epoch in range(1, epochs+1):
    start = np.random.randint(1, train_max_start + 1)

    inp = (np.arange(start, start + 50) / scale).reshape(50, 1)
    tar = (start + 50) / scale

    y_hat = rnn.forward(inp)
    loss = 0.5 * (y_hat - tar) ** 2
    e = y_hat - tar
    rnn.backward(e, learning_rate=learning_rate)

    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | loss={loss:.3e} | pred={y_hat*scale:.2f} | target={tar*scale:.2f}")
        
test_start = 101  # [101..150] -> 151
test_inp = (np.arange(test_start, test_start + 50) / scale).reshape(50, 1)
test_tar = (test_start + 50) / scale

test_pred = rnn.forward(test_inp)

print("\nTest:")
print(f"Input: {test_start}..{test_start+49}  Target: {test_tar*scale:.2f}  Pred: {test_pred*scale:.2f}")
