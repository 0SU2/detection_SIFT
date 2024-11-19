import cv2
import argparse
import pickle
import matplotlib.pyplot as plt

# Resize images to a similar dimension
# This helps improve accuracy and decreases unnecessarily high number of keypoints
def imageResizeTest(image):
    maxD = 1024
    height,width,channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

# Fetch Keypoints and Descriptors from stored files
def fetchKeypointFromFile(image1, angle1):
    filepath = f"./coil-100/objeto{image1}/keypoints/obj{image1}__{angle1}.pkl"
    keypoint = []
    file = open(filepath,'rb')
    deserializedKeypoints = pickle.load(file)
    file.close()
    for point in deserializedKeypoints:
        temp = cv2.KeyPoint(
            x=point[0][0],
            y=point[0][1],
            size=point[1],
            angle=point[2],
            response=point[3],
            octave=point[4],
            class_id=point[5]
        )
        keypoint.append(temp)
    return keypoint

def fetchDescriptorFromFile(image1, angle1):
    filepath = f"./coil-100/objeto{image1}/descriptors/obj{image1}__{angle1}.pkl"
    file = open(filepath,'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor

def calculateResultsFor(image1,image2, angle1, angle2):
    keypoint1 = fetchKeypointFromFile(image1, angle1)
    descriptor1 = fetchDescriptorFromFile(image1, angle1)
    keypoint2 = fetchKeypointFromFile(image2, angle2)
    descriptor2 = fetchDescriptorFromFile(image2, angle2)
    matches = calculateMatches(descriptor1, descriptor2)

    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    plot = getPlotFor(image1, angle1, image2, angle2,keypoint1,keypoint2,matches)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda x: x.distance)
    umbral_distance = 50
    TP = FP = FN = 0
    for match in matches:
        if match.distance < umbral_distance:
            TP += 1
        else:
            FP += 1
    FN = len(keypoint1) + len(keypoint2) - TP - FP
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    accuracy = TP / (TP + FP + FN)   
    print(f"TP (True Positives): {TP}")
    print(f"FP (False Positives): {FP}")
    print(f"FN (False Negatives): {FN}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(score)
    
    plt.imshow(plot),plt.show()

def getPlotFor(image1, angle1, image2, angle2, keypoint1, keypoint2, matches):
    image1 = imageResizeTest(cv2.imread(f"./coil-100/objeto{image1}/images/obj{image1}__{angle1}.png"))
    image2 = imageResizeTest(cv2.imread(f"./coil-100/objeto{image2}/images/obj{image2}__{angle2}.png"))
    return getPlot(image1,image2,keypoint1,keypoint2,matches)

def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))

bf = cv2.BFMatcher()
def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])
            
    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(
        image1,
        keypoint1,
        image2,
        keypoint2,
        matches,
        None,
        [255,255,255],
        flags=2
    )
    return matchPlot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_one', type=str, help='Number of the first image to compare, example: obj1')
    parser.add_argument('--object_two', type=str, help='Number of the second image to compare, exmple: obj2')
    parser.add_argument('--object_one_angle', type=str, default=0, help='Angle of the first image, example: 15')
    parser.add_argument('--object_two_angle', type=str, default=0, help='Angle of the second image, example: 355')
    opt = parser.parse_args()
    calculateResultsFor(opt.object_one, opt.object_two, opt.object_one_angle, opt.object_two_angle)