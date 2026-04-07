print("JANASREE 24BAD040")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"C:\Users\janas\Downloads\Mall_Customers.csv")

df = df.dropna()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

aic_values = []
bic_values = []
log_likelihood_values = []
components_range = range(1, 11)

for n in components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    aic_values.append(gmm.aic(X_scaled))
    bic_values.append(gmm.bic(X_scaled))
    log_likelihood_values.append(gmm.score(X_scaled))

plt.figure()
plt.plot(components_range, aic_values, marker='o', label='AIC')
plt.plot(components_range, bic_values, marker='s', label='BIC')
plt.title("AIC and BIC for Different Number of Components")
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(components_range, log_likelihood_values, marker='^', label='Log-Likelihood')
plt.title("Log-Likelihood for Different Number of Components")
plt.xlabel("Number of Components")
plt.ylabel("Average Log-Likelihood")
plt.legend()
plt.grid(True)
plt.show()

gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X_scaled)

gmm_labels = gmm.predict(X_scaled)
gmm_probabilities = gmm.predict_proba(X_scaled)

df['GMM_Cluster'] = gmm_labels

sil_score_gmm = silhouette_score(X_scaled, gmm_labels)
print("Silhouette Score (GMM):", sil_score_gmm)
print("Log-Likelihood:", gmm.score(X_scaled))
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))

print("\nFirst 10 Cluster Probabilities:")
print(pd.DataFrame(gmm_probabilities[:10]))

plt.figure()
for i in range(5):
    plt.scatter(
        X_scaled[gmm_labels == i, 0],
        X_scaled[gmm_labels == i, 1],
        label=f'Cluster {i}'
    )
plt.title("GMM Clustering (Scaled Data)")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for i in range(5):
    plt.scatter(
        df[df['GMM_Cluster'] == i]['Annual Income (k$)'],
        df[df['GMM_Cluster'] == i]['Spending Score (1-100)'],
        label=f'Cluster {i}'
    )
plt.title("GMM Customer Segmentation (Original Data)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()

x = np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 300)
y = np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 300)
X_grid, Y_grid = np.meshgrid(x, y)
XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X_grid.shape)

plt.figure()
plt.contour(X_grid, Y_grid, Z, levels=15)
for i in range(5):
    plt.scatter(
        X_scaled[gmm_labels == i, 0],
        X_scaled[gmm_labels == i, 1],
        label=f'Cluster {i}'
    )
plt.title("GMM Contour Plot with Clusters")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for i in range(5):
    plt.hist(
        gmm_probabilities[:, i],
        bins=10,
        alpha=0.6,
        label=f'Cluster {i}'
    )
plt.title("Cluster Probability Distribution")
plt.xlabel("Cluster Probability")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

plt.figure()
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=kmeans_labels,
    label='K-Means Clusters'
)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=gmm_labels,
    label='GMM Clusters'
)
plt.title("GMM Clustering")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid(True)
plt.show()

print("\nCluster-wise Mean Values (GMM):")
print(df.groupby('GMM_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
