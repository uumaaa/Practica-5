from Segmentation import k_means
import cv2 as cv

if __name__ == "__main__":
    #Se lee la imagen de los circulos
    img = cv.imread("balls.png")
    #Se convierte a RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #Se crea el objeto de lista de pixeles
    pixels = k_means.PixelList(img)

    #Se obtiene la lista de pixeles
    data = pixels.get_data()

    #Se crea el objeto de clasificador k-means
    kmeans = k_means.Kmeans_classifier(data,7)

    kmeans.fit(max_iterations=1000)

    #Se muestra el resultado de la clasificación
    kmeans.visualize_clusters()