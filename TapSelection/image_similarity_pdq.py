import pdqhash
import cv2
import os
import numpy as np

class image_comparator_pdq:
    
    def __init__(self):
        self.threshold = 15000
        
    def compute_pdq_hash(self, image):
        #hash_vector, quality = pdqhash.compute(image)
        
        '''
        Get all the rotations and flips in one pass.
        hash_vectors is a list of vectors in the following order
        - Original
        - Rotated 90 degrees
        - Rotated 180 degrees
        - Rotated 270 degrees
        - Flipped vertically
        - Flipped horizontally
        - Rotated 90 degrees and flipped vertically
        - Rotated 90 degrees and flipped horizontally
        '''
        #hash_vectors, quality = pdqhash.compute_dihedral(image)

        # Get the floating point values of the hash.
        hash_vector_float, quality = pdqhash.compute_float(image)
        
        return hash_vector_float
    
    def dissimilarity(self, image1, image2):
        #return value of dissimilarity metric
        emb1 = self.compute_pdq_hash(image1)
        emb2 = self.compute_pdq_hash(image2)

        dist = np.square(np.subtract(emb1,emb2)).mean()
        #dist = np.count_nonzero(emb1!=emb2)
        return dist
    
    def is_same(self, image1, image2):
        dis = self.dissimilarity(image1, image2)
        return (dis <= self.threshold, dis)
    
    
if __name__ == "__main__":
    image1 = cv2.imread("./media/oc_5.jpg")
    #image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread("./media/oc_3.jpg")
    #image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    ic = image_comparator_pdq()
    print(ic.is_same(image1, image2))