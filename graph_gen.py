from PIL import Image
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_image_in_lab(image_path):
    # Cargar la imagen
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150, 150))  # Reducir tamaño para acelerar el procesamiento
    img_data = np.array(img) / 255.0  # Normalizar valores RGB a [0, 1]
    
    # Convertir la imagen al espacio LAB
    lab_image = rgb2lab(img_data)
    
    # Extraer las coordenadas L*, a*, b*
    L = lab_image[:, :, 0].flatten()
    a = lab_image[:, :, 1].flatten()
    b = lab_image[:, :, 2].flatten()
    
    # Graficar en 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(a, b, L, c=img_data.reshape(-1, 3), s=1)
    
    # Etiquetas y límites
    ax.set_xlabel('a* (verde ↔ rojo)')
    ax.set_ylabel('b* (azul ↔ amarillo)')
    ax.set_zlabel('L* (luminosidad)')
    ax.set_title('Píxeles de la imagen en el espacio LAB')
    plt.show()


def find_dominant_colors(image_path, n_clusters=8):
    # Cargar la imagen
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150, 150))  # Reducir tamaño para acelerar el procesamiento
    img_data = np.array(img) / 255.0  # Normalizar valores RGB a [0, 1]
    
    # Convertir la imagen al espacio LAB
    lab_data = rgb2lab(img_data)
    pixels = lab_data.reshape(-1, 3)  # Reestructurar a (num_pixels, 3)
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    
    # Obtener los colores representativos en LAB
    cluster_centers = kmeans.cluster_centers_
    
    # Convertir los colores de LAB a RGB para visualización
    rgb_colors = lab2rgb(cluster_centers[np.newaxis, :, :])[0]
    
        # Calcular proporciones de los clusters
    labels = kmeans.labels_
    proportions = np.bincount(labels) / len(labels)
    
    # Graficar los colores como un gráfico de torta
    plt.figure(figsize=(8, 6))
    plt.title(f"{n_clusters} colores más representativos (proporciones)")
    plt.pie(
        proportions,
        colors=rgb_colors,
        labels=[f"Cluster {i}" for i in range(n_clusters)],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.axis("equal")  # Asegura que sea un círculo perfecto
    plt.show()

    return rgb_colors, proportions

#plot_image_in_lab("test.png")
#find_dominant_colors("test.png")