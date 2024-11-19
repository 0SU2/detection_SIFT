import os
import cv2
import pickle
import argparse

# Resize images to a similar dimension
# This helps improve accuracy and decreases unnecessarily high number of keypoints
def imageResizeTrain(image):
    maxD = 1024
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

def computeSIFT(image, sift):
    return sift.detectAndCompute(image, None)

def train(archivos_imagen, carpeta):
  '''
  Train process to obtain all the keypoints and descriptors
  with the SIFT metod.
  '''
  # We use grayscale images for generating keypoints
  imagesBW = []
  for imageName in archivos_imagen:
    imagePath = f"{carpeta}/images/" + str(imageName)
    imagesBW.append(imageResizeTrain(cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)))

  # start with the SIFT method
  sift = cv2.SIFT_create()

  # When using SIFT, this takes a lot of time to compute.
  # Thus, it is suggested, you store the values once computed
  keypoints = []
  descriptors = []
  for i,image in enumerate(imagesBW):
      print("Starting for image: " + archivos_imagen[i])
      keypointTemp, descriptorTemp = computeSIFT(image, sift)
      keypoints.append(keypointTemp)
      descriptors.append(descriptorTemp)
      print("  Ending for image: " + archivos_imagen[i])

  # serialized keypoints
  for i,keypoint in enumerate(keypoints):
      deserializedKeypoints = []
      filepath = f"{carpeta}/keypoints/" + str(archivos_imagen[i].split('.')[0]) + ".pkl"
      for point in keypoint:
          temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
          deserializedKeypoints.append(temp)
      with open(filepath, 'wb') as fp:
          pickle.dump(deserializedKeypoints, fp) 

  for i,descriptor in enumerate(descriptors):
      filepath = f"{carpeta}/descriptors/" + str(archivos_imagen[i].split('.')[0]) + ".pkl"
      with open(filepath, 'wb') as fp:
          pickle.dump(descriptor, fp)
  
  print(f"\ndone :)")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--number_object', type=str)
  opt = parser.parse_args()

  carpeta = f"./coil-100/objeto{opt.number_object}"
  carpeta_images = f"./coil-100/objeto{opt.number_object}/images"
  archivos = os.listdir(carpeta_images)

  archivos_imagen = [archivo for archivo in os.listdir(carpeta_images) if archivo.endswith('.png')]
  train(archivos_imagen, carpeta)