import classicalsim as sim
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#%% Generating training data

# Two qubits first
data_pts = int(1e2 + 1)
g_vals = np.linspace(-2, 2, data_pts)
q_bits = 10
E0 = []
training_data = np.zeros((data_pts, 2))

# generate a 2 qubit hamiltonian, H, for a given magnetic field value, g
# determine ground state energy expectation Tr(\rho_0 H)
# repeat for each value of g
for i in range(len(g_vals)):
    new_hamiltonian = sim.Hamiltonian(q_bits, g_vals[i])
    exp_min_energy = new_hamiltonian.exp_ground
    E0.append(exp_min_energy)
    training_data[i, 1] = g_vals[i]
    if g_vals[i] == 0:
        print(exp_min_energy)

training_data[:, 0] = q_bits
print(training_data.shape)

train_g = training_data[:,1].reshape((data_pts, 1))
train_E0 = np.array(E0).reshape((data_pts, 1))
print(train_g.shape)

plt.figure()
plt.plot(g_vals, E0)
plt.xlabel('Field Coupling Coefficient, g')
plt.ylabel(r'$\langle \psi_0\vert\hat H\vert\psi_0\rangle$')
plt.title(r'1D Ising $\hat{H}$ for ' + str(q_bits) + ' qubits')

X_train, X_test, y_train, y_test = train_test_split(train_g, train_E0, test_size=0.2, random_state=42)

#%% Make feature map using random fourier series

# Approximate the feature map of an RBF kernel by Monte Carlo approximation
rbf_feature_map = RBFSampler(gamma=1, random_state=42)

#%% LASSO Regression Model

# Create a pipeline that will create random fourier features, and then apply LASSO
lasso = Lasso(alpha=0.001)
lasso_pipe = make_pipeline(rbf_feature_map, lasso)

# train model
lasso_pipe.fit(X_train, y_train)


#%% Make Predictions

# check score
print(lasso_pipe.score(X_test, y_test))
# make predictions using X_test data inputs
pred_y = lasso_pipe.predict(X_test)
# plot test data
plt.scatter(X_test, pred_y, marker='x')

plt.savefig('testplot')