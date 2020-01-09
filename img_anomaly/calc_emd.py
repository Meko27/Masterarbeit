# Script to calculate the EMD of two HOGs and a ground distance

import cv2

def calc_EMD(sig1,sig2,cost_mat):

    dist = cv2.EMD(sig1,sig2,cost=cost_mat)

    return dist