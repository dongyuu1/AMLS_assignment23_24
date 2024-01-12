from skimage import feature
import numpy as np
import cv2
from tqdm import trange


class FeatureExtractor:
    """
    This class is for extracting features of input data
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def extract_hog_features(self, img_data):
        """
        Extract the Histogram of Oriented Gradients of input images
        :param img_data: The input images
        :return: Extracted HOG features for images
        """
        # The batch size of the input
        b = img_data.shape[0]
        feats = []

        for index in trange(b):
            img = img_data[index]
            # Resize the image to make it fit the custom parameter setting of HOG feature extraction
            img = cv2.resize(img, self.cfg.FEAT.RESIZE)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            feat = feature.hog(img_gray,
                               pixels_per_cell=self.cfg.FEAT.HOG.PIXELS_PER_CELL,
                               cells_per_block=self.cfg.FEAT.HOG.CELLS_PER_BLOCK,
                               feature_vector=True)
            feats.append(feat)
        # Stack the features together
        feats = np.stack(feats)
        return feats

    def extract_lbp_features(self, img_data):
        """
        Extract the Local Binary Pattern features of input images
        :param img_data: The input images
        :return: Extracted HOG features for images
        """
        b = img_data.shape[0]
        feats = []

        for index in trange(b):
            img = img_data[index]
            # Resize to make the image fit the custom parameter setting of LBP feature extraction
            img = cv2.resize(img, self.cfg.FEAT.RESIZE)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            feat = feature.local_binary_pattern(img_gray,
                                                P=self.cfg.FEAT.LBP.P,
                                                R=self.cfg.FEAT.LBP.R)
            feats.append(feat.flatten())
        # Stack the features together
        feats = np.stack(feats)
        return feats

    def extract_hog_and_lbp_features(self, img_data):
        """
        Extract the composed HOG and LBP features of input images
        :param img_data: The input images
        :return: Extracted composed features for images
        """
        b = img_data.shape[0]
        feats = []

        for index in trange(b):
            img = img_data[index]
            # Resize to make the image fit the custom parameter setting of LBP feature extraction
            img = cv2.resize(img, self.cfg.FEAT.RESIZE)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            lbp_feat = feature.local_binary_pattern(img_gray,
                                                P=self.cfg.FEAT.LBP.P,
                                                R=self.cfg.FEAT.LBP.R)
            hog_lbp_feat = feature.hog(lbp_feat,
                               pixels_per_cell=self.cfg.FEAT.HOG.PIXELS_PER_CELL,
                               cells_per_block=self.cfg.FEAT.HOG.CELLS_PER_BLOCK,
                               feature_vector=True)
            feats.append(hog_lbp_feat)
        # Stack the features together
        feats = np.stack(feats)
        return feats

#fe = FeatureExtractor()
#data = np.zeros((278, 28, 28))
#data = fe.extract_lbp_features(data)
