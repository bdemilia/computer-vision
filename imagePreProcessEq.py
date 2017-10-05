
###
def draw_hsv(flow):
   h, w = flow.shape[:2]
   fx, fy = flow[:,:,0], flow[:,:,1]
   ang = np.arctan2(fy, fx) + np.pi
   v = np.sqrt(fx*fx+fy*fy)
   hsv = np.zeros((h, w, 3), np.uint8)
   hsv[...,0] = ang*(180/np.pi/2)
   hsv[...,1] = 255
   hsv[...,2] = np.minimum(v*4, 255)
   bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
return bgr

def run_check_flow():

   folder ='test_hq1024x1024'
   #folder ='test512x512'
   #folder ='test'
   #folder ='test_hq'
   #name = 'a67333bd4065_09'  #'0c56c040a690_09'
   name = '1b218148cb02_09'  #'0c56c040a690_09'

   img_file = DIR + '/images/%s/%s.jpg'%(folder,name)
   image = cv2.imread(img_file)

   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   height,width,C = image.shape
   dx =np.zeros((height,width,1),np.float32)
   dy =np.zeros((height,width,1),np.float32)

   dx[:,1:,0] = gray[:,1:]-gray[:,:-1]
   dy[1:,:,0] = gray[1:]-gray[:-1]

   flow = np.dstack((dx,dy))

   f = draw_hsv(flow)
   cv2.imwrite('folder_create',f)
   im_show('flow',  f,  resize=1)
   im_show('dx',  np.abs(dx),  resize=1)
   im_show('dy',  np.abs(dy),  resize=1)
   cv2.waitKey(0)
