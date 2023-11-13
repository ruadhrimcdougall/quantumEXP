import classicalsim as sim
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

#%% Generating training data

# Two qubits first
data_pts = int(1e3 + 1)
g_vals = np.linspace(-2, 2, data_pts)
q_bits = 2
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

# plt.figure()
# plt.plot(g_vals, E0)
# plt.xlabel('Field Coupling Coefficient, g')
# plt.ylabel(r'$\langle \psi_0\vert\hat H\vert\psi_0\rangle$')
# plt.title(r'1D Ising $\hat{H}$ for ' + str(q_bits) + ' qubits')

# X_train, X_test, y_train, y_test = train_test_split(train_g, train_E0, test_size=0.2, random_state=42)
# if using learning_curve to do the splitting,
X_train = train_g
y_train = train_E0

#%% Make feature map using random fourier series

# Approximate the feature map of an RBF kernel by Monte Carlo approximation
rbf_feature_map = RBFSampler(gamma=1, random_state=42)

#%% LASSO Regression Model

# Create a pipeline that will create random fourier features, and then apply LASSO
lasso = Lasso(alpha=0.001, tol=0.0005)
lasso_pipe = make_pipeline(rbf_feature_map, lasso)
# train model
# lasso_pipe.fit(X_train, y_train)

# use the learning curve feature to compare training of different data set size
train_sizes, train_scores, test_scores = learning_curve(lasso_pipe, X_train, y_train, train_sizes=np.linspace(0.6, 1.0, 10), cv=2)

print(train_scores.shape)
train_mean = np.mean(train_scores, axis=1)
print(train_mean.shape)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

#%% Make Predictions

# Plotting the learning curves
plt.figure()
# plt.yscale("log") 
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

plt.title("Learning Curve - 2 Qubits")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.savefig('lasso_learning_curve')

# # check score
# print(lasso_pipe.score(X_test, y_test))
# # make predictions using X_test data inputs
# pred_y = lasso_pipe.predict(X_test)
# # plot test data
# plt.scatter(X_test, pred_y, marker='x')
# plt.savefig('testplot')