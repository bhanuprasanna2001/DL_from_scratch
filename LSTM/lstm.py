import numpy as np
np.random.seed(42)

# The initial conditions are h_0, C_0, where they are just 0s. AWESOME.
# Full scratch implementation of LSTM and training to predict the next number.
# 1,2,3,4,5 -> Model must predict 6. This is short sequence, will train with large seq.

class LSTM:
    
    def __init__(self, input_dim=1, hidden_dim=100, seq_len=50):
        self.input_dim = input_dim                                                                  # D
        self.hidden_dim = hidden_dim                                                                # H
        self.output_dim = input_dim                                                                 # O
        self.seq_len = seq_len                                                                      # T
        
        # Forget Gate
        self.W_f = np.random.randn(self.hidden_dim, self.hidden_dim + self.input_dim) * 0.01        # H x (H + D)
        self.b_f = np.random.randn(self.hidden_dim, 1) * 0.01                                       # H x 1
        
        # Input Gate
        self.W_i = np.random.randn(self.hidden_dim, self.hidden_dim + self.input_dim) * 0.01        # H x (H + D)
        self.b_i = np.random.randn(self.hidden_dim, 1) * 0.01                                       # H x 1
        
        # Candidate Gate
        self.W_g = np.random.randn(self.hidden_dim, self.hidden_dim + self.input_dim) * 0.01        # H x (H + D)
        self.b_g = np.random.randn(self.hidden_dim, 1) * 0.01                                       # H x 1
        
        # Output Gate
        self.W_o = np.random.randn(self.hidden_dim, self.hidden_dim + self.input_dim) * 0.01        # H x (H + D)
        self.b_o = np.random.randn(self.hidden_dim, 1) * 0.01                                       # H x 1
        
        # Output Prediction
        self.W_y = np.random.randn(self.output_dim, self.hidden_dim) * 0.01                         # O x H
        self.b_y = np.random.randn(self.output_dim, 1) * 0.01                                       # O x 1
        
        
    def forward(self, x):
        self.x = x.copy()
        
        # Hidden State and Candidate vectors
        self.h = np.zeros((self.seq_len + 1, self.hidden_dim, 1))                                   # (T + 1) x H x 1
        self.C = np.zeros((self.seq_len + 1, self.hidden_dim, 1))                                   # (T + 1) x H x 1
        self.u = np.zeros((self.seq_len + 1, self.hidden_dim, 1))                                   # (T + 1) x H x 1
        
        # Store z, f, i, g, o
        self.z = np.zeros((self.seq_len, self.hidden_dim + self.input_dim, 1))                      # T x (H + D) x 1
        self.f = np.zeros((self.seq_len, self.hidden_dim, 1))
        self.i = np.zeros((self.seq_len, self.hidden_dim, 1))
        self.g = np.zeros((self.seq_len, self.hidden_dim, 1))
        self.o = np.zeros((self.seq_len, self.hidden_dim, 1))
        
        # self.print_shapes()
        
        for t in range(1, self.seq_len + 1):
            # 1. Form concatenated vector:
            # print(f"h_t shape: {self.h[t-1].shape}, x_t shape: {self.x[t-1].shape}")
            self.z[t-1] = np.concat((self.h[t-1], self.x[t-1]), axis=0)                               # (H + D) x 1
            # print(f"z shape: {self.z[t-1].shape}")
            
            # 2. Forget Gate:
            f_pre = self.W_f @ self.z[t-1] + self.b_f                                                 # H x 1
            self.f[t-1] = self.sigmoid(f_pre)                                                         # H x 1
            # print(f"f shape: {self.f[t-1].shape}")
            
            # 3. Input Gate:
            i_pre = self.W_i @ self.z[t-1] + self.b_i                                                 # H x 1
            self.i[t-1] = self.sigmoid(i_pre)                                                         # H x 1
            # print(f"i shape: {self.i[t-1].shape}")
            
            # 4. Candidate Content
            g_pre = self.W_g @ self.z[t-1] + self.b_g                                                 # H x 1
            self.g[t-1] = np.tanh(g_pre)                                                              # H x 1
            # print(f"g shape: {self.g[t-1].shape}")
            
            # 5. Cell State Update
            self.C[t] = np.multiply(self.f[t-1], self.C[t-1]) + np.multiply(self.i[t-1], self.g[t-1]) # H x 1
            self.u[t] = np.tanh(self.C[t])
            # print(f"C shape: {self.C[t].shape}")
            
            # 6. Output gate
            o_pre = self.W_o @ self.z[t-1] + self.b_o                                                 # H x 1
            self.o[t-1] = self.sigmoid(o_pre)                                                         # H x 1
            # print(f"o shape: {self.o[t-1].shape}")
            
            # 7. Hidden State
            self.h[t] = np.multiply(self.o[t-1], self.u[t])                                           # H x 1
            # print(f"h shape: {self.h[t].shape}")
            
        # Now Output Prediction
        self.y_hat = self.W_y @ self.h[self.seq_len] + self.b_y                                       # 1 x 1
        # print(f"y_hat shape: {self.y_hat.shape}")
        
        return self.y_hat
    
    
    def backward(self, y_true, y_pred, lr=0.001):
        
        G_wf = G_bf = G_wi = G_bi = G_wg = G_bg = G_wo = G_bo = G_wy = G_by = 0
        
        e = y_pred - y_true
        
        G_wy = e @ self.h[self.seq_len].T                                                           # 1 x H
        G_by = e                                                                                    # 1
        # print(f"G_wy shape: {G_wy.shape}, G_by shape: {G_by.shape}")
        
        G_h = self.W_y.T @ e                                                                        # H x 1
        G_C = np.zeros((self.hidden_dim, 1))                                                        # H x 1
        # print(f"G_h shape: {G_h.shape}, G_C shape: {G_C.shape}")
        
        for t in range(self.seq_len, 0, -1):
            # 1. Output Gate Gradient
            G_o = np.multiply(G_h, self.u[t])                                                       # H x 1
            # print(f"G_o shape: {G_o.shape}")
            
            # 2. Add to cell Gradient
            G_C = G_C + np.multiply(np.multiply(G_h, self.o[t-1]), (1 - (self.u[t] ** 2)))         # H x 1
            # print(f"G_C shape: {G_C.shape}")
            
            # 3. Forget Gate Gradient
            G_f = np.multiply(G_C, self.C[t-1])                                                     # H x 1
            # print(f"G_f shape: {G_f.shape}")
            
            # 4. Previous Cell Gradient
            G_C_1 = np.multiply(G_C, self.f[t-1])                                                   # H x 1
            # print(f"G_C_1 shape: {G_C_1.shape}")
            
            # 5. Input Gate Gradient
            G_i = np.multiply(G_C, self.g[t-1])                                                     # H x 1
            # print(f"G_i shape: {G_i.shape}")
            
            # 6. Candidate Gradient
            G_g = np.multiply(G_C, self.i[t-1])                                                     # H x 1
            # print(f"G_g shape: {G_g.shape}")
            
            # 7. Sigmoid preactivation gradients
            G_f_pre = np.multiply(np.multiply(G_f, self.f[t-1]), (1 - self.f[t-1]))                # H x 1
            G_i_pre = np.multiply(np.multiply(G_i, self.i[t-1]), (1 - self.i[t-1]))                # H x 1
            G_o_pre = np.multiply(np.multiply(G_o, self.o[t-1]), (1 - self.o[t-1]))                # H x 1
            # print(f"G_f_pre shape: {G_f_pre.shape}")
            # print(f"G_i_pre shape: {G_i_pre.shape}")
            # print(f"G_o_pre shape: {G_o_pre.shape}")
            
            # 8. Candidate tanh preactivation gradient
            G_g_pre = np.multiply(G_g, (1 - (self.g[t-1] ** 2)))                                    # H x 1
            # print(f"G_g_pre shape: {G_g_pre.shape}")
            
            # 9. Accumulate weight and bias gradients
            G_wf += G_f_pre @ self.z[t-1].T; G_bf += G_f_pre                                        # H x (H+D), H x 1
            G_wi += G_i_pre @ self.z[t-1].T; G_bi += G_i_pre                                        # H x (H+D), H x 1
            G_wg += G_g_pre @ self.z[t-1].T; G_bg += G_g_pre                                        # H x (H+D), H x 1
            G_wo += G_o_pre @ self.z[t-1].T; G_bo += G_o_pre                                        # H x (H+D), H x 1
            # print(f"G_wf shape: {G_wf.shape}, G_bf shape: {G_bf.shape}")
            # print(f"G_wi shape: {G_wi.shape}, G_bi shape: {G_bi.shape}")
            # print(f"G_wg shape: {G_wg.shape}, G_bg shape: {G_bg.shape}")
            # print(f"G_wo shape: {G_wo.shape}, G_bo shape: {G_bo.shape}")
            
            # 10. Gradient w.r.t. concatenated vector z
            G_z = self.W_f.T @ G_f_pre + self.W_i.T @ G_i_pre + self.W_g.T @ G_g_pre + self.W_o.T @ G_o_pre  # (H+D) x 1
            # print(f"G_z shape: {G_z.shape}")
            
            # 11. Split gradient for hidden state and input
            G_h = G_z[:self.hidden_dim]                                                             # H x 1
            # print(f"G_h (next) shape: {G_h.shape}")
            
            # 12. Update cell gradient for next iteration
            G_C = G_C_1                                                                             # H x 1
            # print(f"G_C (next) shape: {G_C.shape}")
        
        # Clip gradients to prevent exploding gradients
        for grad in [G_wf, G_bf, G_wi, G_bi, G_wg, G_bg, G_wo, G_bo, G_wy, G_by]:
            np.clip(grad, -5, 5, out=grad)
        
        # Update weights and biases
        self.W_f -= lr * G_wf
        self.b_f -= lr * G_bf
        self.W_i -= lr * G_wi
        self.b_i -= lr * G_bi
        self.W_g -= lr * G_wg
        self.b_g -= lr * G_bg
        self.W_o -= lr * G_wo
        self.b_o -= lr * G_bo
        self.W_y -= lr * G_wy
        self.b_y -= lr * G_by
            

    @staticmethod
    def sigmoid(matx):
        return 1 / (1 + np.exp(-matx))
        
    def print_shapes(self):
        # Print all weights and biases (shapes only)
        print("=" * 50)
        print("LSTM Parameters:")
        print("=" * 50)
        print(f"\nForget Gate:")
        print(f"W_f shape: {self.W_f.shape}")
        print(f"b_f shape: {self.b_f.shape}")

        print(f"\nInput Gate:")
        print(f"W_i shape: {self.W_i.shape}")
        print(f"b_i shape: {self.b_i.shape}")

        print(f"\nCandidate Gate:")
        print(f"W_g shape: {self.W_g.shape}")
        print(f"b_g shape: {self.b_g.shape}")

        print(f"\nOutput Gate:")
        print(f"W_o shape: {self.W_o.shape}")
        print(f"b_o shape: {self.b_o.shape}")

        print(f"\nOutput Prediction:")
        print(f"W_y shape: {self.W_y.shape}")
        print(f"b_y shape: {self.b_y.shape}")

        print(f"\nHidden and Cell States:")
        print(f"h shape: {self.h.shape}")
        print(f"C shape: {self.C.shape}")
        print("=" * 50)
        

lstm = LSTM(input_dim=1, hidden_dim=100, seq_len=250)

scale = 1000.0
train_max_start = 700

num_epochs = 5000
learning_rate = 0.05

for epoch in range(1, num_epochs + 1):
    start = np.random.randint(1, train_max_start + 1)
    
    inp = (np.arange(start, start + 250) / scale).reshape(250, 1, 1)
    y_true = np.array([(start + 250) / scale]).reshape(1, 1)
    
    y_pred = lstm.forward(inp)
    loss = 0.5 * (y_pred - y_true) ** 2
    
    lstm.backward(y_true, y_pred, lr=learning_rate)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss[0,0]:.3e} | Pred: {y_pred[0,0]*scale:.2f} | Target: {y_true[0,0]*scale:.2f}")

# Test on unseen data
test_start = 101
test_inp = (np.arange(test_start, test_start + 250) / scale).reshape(250, 1, 1)
test_tar = np.array([(test_start + 250) / scale]).reshape(1, 1)

test_pred = lstm.forward(test_inp)

print(f"\nTest:")
print(f"Input: {test_start}..{test_start+249}  Target: {test_tar[0,0]*scale:.2f}  Pred: {test_pred[0,0]*scale:.2f}")
