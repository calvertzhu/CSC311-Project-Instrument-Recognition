#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# In[2]:


# Load files
X = np.load("X.npy") # features, shape: (6705, 128, 128) 
y = np.load("y.npy") # labels, shape: (6705,)


# In[3]:


print("X shape:", X.shape)
print("y shape:", y.shape)


# In[4]:


# Flattening for kNN
X_flat = X.reshape((X.shape[0], -1))  # shape: (6705, 16384)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling to prevent bias
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train kNN Classifier
knn = KNeighborsClassifier(n_neighbors = 5) # <--Tune this 
knn.fit(X_train_scaled, y_train)

# Predict & Evaluate
y_pred = knn.predict(X_test_scaled)

print("=== Classification Report ===\n")
print(classification_report(y_test, y_pred))


# In[ ]:




