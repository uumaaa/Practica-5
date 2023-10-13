from Segmentation import k_means
import matplotlib.pyplot as plt

import cv2 as cv

if __name__ == "__main__":

    img = cv.imread("balls.png")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    pixels = k_means.PixelList(img)

    data = pixels.get_data()
    kmeans = k_means.Kmeans_classifier(data,7)
    #Se crea el objeto de clasificador k-means
    kmeans.fit(max_iterations=1000)
    #Se muestra el resultado de la clasificación
    kmeans.visualize_clusters()
    # wcss_values = k_means.Kmeans_classifier.calculate_wcss(data, 10)  
    # #^ Se muestra la gráfica del método del codo ^#
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, len(wcss_values) + 1), wcss_values, marker='o', color='purple')
    # plt.title('Método del Codo')
    # plt.xlabel('Número de Clústeres (k)')
    # plt.ylabel('WCSS')
    # plt.grid()
    # plt.show()


    #Se muestran las imágenes segmentadas
    cluster_images = kmeans.generate_cluster_images(img)

    #Se muestran las imágenes segmentadas
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    for i in range(7):
        plt.subplot(2, 4, i + 2)
        plt.imshow(cluster_images[i])
        plt.title(f'Cluster {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

  
    


