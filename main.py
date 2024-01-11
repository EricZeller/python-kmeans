import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# Eingabe der Daten
def user_input_data():
    num_samples = int(input("Geben Sie die Anzahl der Datenpunkte ein: "))
    daten = []
    labels = []

    for i in range(num_samples):        # Schleife für Eingabe von i Datensätzen
        label = input(f"Geben Sie das Label für den Datenpunkt {i + 1} ein: ")
        feature1 = float(input(f"Geben Sie den Wert für Feature 1 vom Datenpunkt {i + 1} ein: "))
        feature2 = float(input(f"Geben Sie den Wert für Feature 2 vom Datenpunkt {i + 1} ein: "))

        daten.append([feature1, feature2])         # Hinzufügen der neuen Daten in die Arrays
        labels.append(label)

    return np.array(daten), labels


# Durchführung K-means-Clusteranalyse mithilfe der Library sklearn
def k_means_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids


# Durchführung hierarchischen Clusteranalyse (Übergabe an Library)
def hierarchical_clustering(data, num_clusters):
    linkage_matrix = linkage(data, 'ward')
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    labels = clustering.fit_predict(data)
    return labels, linkage_matrix


# Funktion zum Erstellen der Diagramme (Übergabe an Library)
def plot_clusters(data, labels, centroids, label_names=None):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='r')
    if label_names is not None:           # Labels neben den Datenpunkten anzeigen
        for i, txt in enumerate(label_names):
            plt.annotate(txt, (data[i, 0], data[i, 1]), xytext=(8, -5), textcoords='offset points')

    plt.title('K-means Clusteranalyse')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Funktion zur Darstellung des Dendrogramms
def plot_dendrogram(linkage_matrix):
    dendrogram(linkage_matrix)
    plt.title('Hierarchisches Dendrogramm')
    plt.xlabel('Index der Datenpunkte')
    plt.ylabel('Abstand')
    plt.show()


# Benutzereingabe und Aufruf zur Datenpunkteingabe
data, true_labels = user_input_data()

# Benutzereingabe für die Anzahl der Cluster
num_clusters = int(input("Geben Sie die Anzahl der Cluster für die K-means Clusteranalyse ein: "))

# K-means-Clusteranalyse durchführen
labels_kmeans, centroids_kmeans = k_means_clustering(data, num_clusters)

# Ergebnisse ausgeben
print("K-means Cluster Labels:")
print(labels_kmeans)

# Darstellung der K-means Cluster
plot_clusters(data, labels_kmeans, centroids_kmeans, label_names=true_labels)

# Benutzereingabe für die Anzahl der Cluster für hierarchische Clusteranalyse
num_clusters_hierarchical = int(input("Geben Sie die Anzahl der Cluster für die hierarchische Clusteranalyse ein: "))

# Hierarchische Clusteranalyse durchführen
labels_hierarchical, linkage_matrix = hierarchical_clustering(data, num_clusters_hierarchical)

# Ergebnisse ausgeben
print("Hierarchische Cluster Labels:")
print(labels_hierarchical)

# Dendrogramm darstellen
plot_dendrogram(linkage_matrix)

# Tabelle mit den Eingaben erstellen und ausgeben
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2'])
df['Label'] = true_labels
print("\nTabelle mit Eingaben:")
print(df)
