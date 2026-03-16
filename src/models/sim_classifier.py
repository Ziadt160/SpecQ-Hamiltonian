import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .pauli_utils import get_pauli_tensor

class SIMClassifier(BaseEstimator, ClassifierMixin):
    """
    Simplified Hamiltonian (SIM) Classifier.
    
    This model implements a quadratic decision boundary by mapping classical input vectors
    to the space of Pauli interaction expectations. Specifically, it computes features
    $\phi_P(x) = x^T P x$ for a predefined set of Pauli strings $P$ and performs 
    regularized linear classification in this interaction space.
    
    Attributes:
        pauli_strings (list of str): The set of observables used for the quadratic feature map.
        classifier (LogisticRegression): The internal linear model trained on Pauli features.
        scaler (StandardScaler): Normalizes interaction features to zero mean and unit variance.
    """
    def __init__(self, pauli_strings=None, C=1.0, random_state=None):
        """
        Args:
            pauli_strings (list of str): List of Pauli strings to use as basis.
            C (float): Regularization parameter for the internal linear classifier.
            random_state (int): Random seed.
        """
        self.pauli_strings = pauli_strings if pauli_strings is not None else ['II']
        self.C = C
        self.random_state = random_state
        self.models_ = {} # Only need one internal model, but keeping structure flexible
        self.classifier = None
        self.scaler = StandardScaler()
        self.pauli_matrices_ = []
        
    def _compute_features(self, X):
        """
        Compute features phi_j(x_i) = x_i^T * P_j * x_i for all samples and Pauli strings.
        
        Args:
            X (np.array): Input data of shape (n_samples, 2^n_qubits) or (n_samples, n_features).
                          Note: Dimensions must match the Pauli tensor size.
                          
        Returns:
            np.array: Feature matrix of shape (n_samples, n_pauli_strings).
        """
        n_samples = X.shape[0]
        n_features = len(self.pauli_strings)
        Phi = np.zeros((n_samples, n_features))
        
        # Precompute matrices if not already done (caching strategy)
        if not self.pauli_matrices_:
            self.pauli_matrices_ = [get_pauli_tensor(s) for s in self.pauli_strings]
            
        # Check dimension consistency
        dim = self.pauli_matrices_[0].shape[0]
        if X.shape[1] != dim:
             raise ValueError(f"Input dimension {X.shape[1]} does not match Pauli dimension {dim}.")

        # Optimization:
        # x^T P x = Trace(P * (x x^T))
        # But for single vector, just simple matrix multiplication is likely fast enough for small N.
        # For larger N, specialized Pauli multiplication is better, but we stick to dense for now.
        
        for j, P in enumerate(self.pauli_matrices_):
            # Compute x^T P x for all x
            # (N, D) @ (D, D) -> (N, D)
            # then dot product with x
            XP = X @ P
            # Element-wise multiply X and XP, then sum over axis 1 = row-wise dot product
            # This computes x_i^T (P x_i) efficiently
            # Note: Result should be real for Hermitian P and real x, but complex could happen with Y
            vals = np.einsum('ij,ij->i', X.conj(), XP) # General case if X is complex
            Phi[:, j] = vals.real # SIM usually assumes real features
            
        return Phi

    def fit(self, X, y):
        """
        Fit the model.
        
        Args:
            X (np.array): Training data.
            y (np.array): Labels.
        """
        # Generate matrices
        self.pauli_matrices_ = [get_pauli_tensor(s) for s in self.pauli_strings]
        
        # Compute features
        Phi = self._compute_features(X)
        
        # Scale features
        Phi_scaled = self.scaler.fit_transform(Phi)
        
        # Train linear classifier
        self.classifier = LogisticRegression(C=self.C, random_state=self.random_state, solver='liblinear')
        self.classifier.fit(Phi_scaled, y)
        
        return self

    def predict(self, X):
        """
        Predict class labels.
        """
        Phi = self._compute_features(X)
        Phi_scaled = self.scaler.transform(Phi)
        return self.classifier.predict(Phi_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        Phi = self._compute_features(X)
        Phi_scaled = self.scaler.transform(Phi)
        return self.classifier.predict_proba(Phi_scaled)

    def get_feature_weights(self):
        """
        Return the learned weights for each Pauli string.
        """
        if self.classifier is None:
            raise RuntimeError("Model not fitted")
        return self.classifier.coef_[0]
