import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import mediapipe as mp
from gfpgan import GFPGANer



def image_detection(img):
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5) as face_detector:
        imported_image = img.copy()
        if imported_image.shape[-1] == 4:
            imported_image = cv2.cvtColor(imported_image, cv2.COLOR_RGBA2RGB)
        results = face_detector.process(imported_image)
        frame_height, frame_width, c = imported_image.shape
        if results.detections:
            l = len(results.detections)
    
            if l == 0 or results.detections == None:
                print("no person is found in the image")
                return -1, -1
            elif l > 1:
                print("multiple persons are found")
                return -1, -1
            for face in results.detections:
                face_react = np.multiply(
                        [
                            face.location_data.relative_bounding_box.xmin,
                            face.location_data.relative_bounding_box.ymin,
                            face.location_data.relative_bounding_box.width,
                            face.location_data.relative_bounding_box.height,
                        ],[frame_width, frame_height, frame_width, frame_height]).astype(int)
                
                cv2.rectangle(imported_image, face_react, color=(255, 255, 255), thickness=2)
                key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
                key_points_coords = np.multiply(key_points,[frame_width,frame_height]).astype(int)
                i = 1
                for p in key_points_coords:
                    cv2.circle(imported_image, p, 4, (255, 255, 255), 2)
                    cv2.circle(imported_image, p, 2, (0, 0, 0), -1)
                    i += 1
                
                return face_react, key_points_coords

            return -1, -1

        return -1, -1




def align_face(image, keypoints):
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center_x, center_y = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle=angle, scale=1.0)
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return aligned_image


def align_detect_image(img, keypoints):
    aligned_image = align_face(img, keypoints)
    # plt.imshow(aligned_image)
    r2, c2 = image_detection(aligned_image)
    # print(r2, c2)
    return aligned_image, r2


from keras_facenet import FaceNet
embedder = FaceNet()



def enhance_faces(input_img):
    # Use the correct variable for input image
    result = gfpganer.enhance(input_img, has_aligned=False, only_center_face=False)
    
    if isinstance(result, tuple):
        restored_image = result[1][0]  # Access the first restored image
        if isinstance(restored_image, np.ndarray):
            # Ensure the restored image has correct data type
            if restored_image.dtype != np.uint8:
                restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
        return restored_image
    else:
        print("Unexpected result format from GFPGAN.")
        return None


base_path = os.getcwd()
# inter_path = os.path.join(base_path, "models");
model_path = os.path.join(base_path, "models\GFPGANv1.4.pth")
gfpganer = GFPGANer(
    model_path=model_path,
    upscale=2,  # Upscaling factor
    arch="clean",  # For general restoration
    channel_multiplier=2  # Default channel multiplier
)



def cosine_similarity(embedding1, embedding2):

    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()


    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm_embedding1 * norm_embedding2)
    return similarity




def get_embedding(img):
    new_img = img.copy()
    results, coords = image_detection(new_img)
    if type(results) == int:
        return -1

    keypoints = {}
    keypoints['left_eye'] = coords[0]
    keypoints['right_eye'] = coords[1]
    
    aligned_image, coords = align_detect_image(new_img, keypoints)
    x, y, h, w = coords
    cropped_img = aligned_image[y:y+h, x:x+w]
    enhanced_img = enhance_faces(cropped_img)
    return embedder.embeddings([enhanced_img])
        




img = plt.imread("enhanced_images/Ashish.jpg")
print(get_embedding(img))






