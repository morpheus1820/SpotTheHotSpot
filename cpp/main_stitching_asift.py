import numpy as np
import math
import cv2
import video
from common import anorm2, draw_str
from time import clock
import itertools as it
from multiprocessing.pool import ThreadPool
from common import Timer
from find_obj import init_feature, filter_matches, explore_match

def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs
    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)
    for i, (k, d) in enumerate(ires):
        print 'affine sampling: %d / %d\r' % (i+1, len(params)),
        keypoints.extend(k)
        descrs.extend(d)
    print
    return keypoints, np.array(descrs)





skip_frames=2

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0


    def run(self):
        # init big image fro stitching
        ret, frame = self.cam.read()
        frame=cv2.resize(frame,(320,240))
        h,w,d=frame.shape
        big_image = np.zeros((h*12,w*3,3), np.uint8)
        starty=h*11
        startx=w
        total_transl_x=0
        total_transl_y=0

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray=frame_gray
        self.prev_frame=frame

        detector, matcher = init_feature('sift-flann')
        pool=ThreadPool(processes = cv2.getNumberOfCPUs())

        while True:
            for i in range(skip_frames):
                ret, frame = self.cam.read()
            ret, frame = self.cam.read()
            frame=cv2.resize(frame,(320,240))

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            img0, img1 = self.prev_gray, frame_gray

            kp1, desc1 = affine_detect(detector, img0[10:h-50,10:w-10], pool=pool)
            kp2, desc2 = affine_detect(detector, img1[10:h-50,10:w-10], pool=pool)
            print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

            with Timer('matching'):
                raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
            p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
            if len(p1) >= 4:
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                print '%d / %d  inliers/matched' % (np.sum(status), len(status))
                # do not draw outliers (there will be a lot of them)
                kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]

                warp = cv2.warpPerspective(img0[10:h-50,10:w-10], H, (w, h*2))
                cv2.imshow("warped",warp)

            else:
                H, status = None, None
                print '%d matches found, not enough for homography estimation' % len(p1)

            vis = explore_match('affine find_obj', img0, img1, kp_pairs, None, H)




            # stitching-----------------------
            translation = np.zeros((3,1))  #3,1
            if len(p1)>4 and len(p2)>4:

                # temp1=[]
                # # temp2=[]

                # for i in range(len(kp1)):
                #     print kp1[i].pt+ (0,)
                #     temp1.append(kp1[i].pt+ (0,))
                # # for i in range(len(kp2)):
                # #     temp2.append(kp2[i].pt)
                # points1.astype(np.uint8)

                # points1 = np.array(temp1)
                # print points1
                # # points2 = np.array(temp2)

                # Hr=cv2.estimateRigidTransform(points1, points1,False)

                translation[:,0] = H[:,2] #Hr[:,2]

                # rotation = np.zeros((3,3))
                # rotation[:,0] = H[:,0]
                # rotation[:,1] = H[:,1]
                # rotation[:,2] = np.cross(H[0:3,0],H[0:3,1])

                # print "x translation:",translation[0]
                # print "y translation:",translation[1]

                draw_str(vis, (20, 40), 'x-axis translation: %.1f' % translation[0])
                draw_str(vis, (20, 60), 'y-axis translation: %.1f' % translation[1])

                if translation[0]<60 and translation[1]<60:  #check for bad H
                    total_transl_x+=int(translation[0])
                    total_transl_y+=int(translation[1])

                    draw_str(vis, (20, 80), 'tot x-axis translation: %.1f' % total_transl_x)
                    draw_str(vis, (20, 100), 'tot y-axis translation: %.1f' % total_transl_y)

                    #h,w,d=frame.shape

                    frame_over=self.prev_frame[10:h-50,10:w-10].copy()
                    overlay = cv2.warpPerspective(frame_over, H, (w, h))
                    frame_h,frame_w,d=frame_over.shape

                    cv2.imshow('overlay',overlay)
                    #vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0.0)
                    big_image[starty-int(total_transl_y):starty-int(total_transl_y)+frame_h,startx-int(total_transl_x):startx-int(total_transl_x)+frame_w]=overlay[0:frame_h,0:frame_w].copy()

            #small_image=big_image.copy()
            big_h,big_w,d=big_image.shape
            small_image=cv2.resize(big_image,(big_w/4,big_h/4))
            cv2.imshow('stitching', small_image)
            #cv2.imwrite("result.jpg",big_image);



            self.frame_idx += 1
            self.prev_gray = frame_gray
            self.prev_frame=frame

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break



def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0

    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
