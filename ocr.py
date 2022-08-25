import tools.infer.predict_rec5_final as ocr
import numpy as np
import cv2


pic = cv2.imread('data/num_data/3_14.jpg')
mat = np.asarray(pic)
print(mat.shape)
B = ocr.ocr(mat)
print(B[0][0])