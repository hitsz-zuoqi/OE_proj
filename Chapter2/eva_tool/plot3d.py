from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

with open('../build/output.poses','r') as f:
    poses = f.readlines()
posexyz = []
for pose in poses:
    pose = np.array([float(s) for s in pose[:-1].split(' ')]).reshape((3,4))
    # Î»ÒÆÏòÁ¿
    dataxyz = pose[:,3]
    # Ðý×ª¾ØÕó
    rot = np.linalg.inv(pose[:,:3])
    posexyz.append(dataxyz.tolist())
posedata_gt = np.array(posexyz)

fig = plt.figure()
ax = fig.gca(projection='3d')
# ÉèÖÃÍ¼ÏñÐÅÏ¢
ax.set_title("3D_Curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
point_num = 480

# ÔÚKITTI²É¼¯µÄÊý¾Ý¼¯ÖÐ£¬zÖáÏòÇ°£¬xÏòÓÒ£¬yÖáÏòÏÂ
figure = ax.plot(posedata_gt[:point_num,2],posedata_gt[:point_num,0],-posedata_gt[:point_num,1],c='r',lw =0.5)
#figure = ax.scatter(posedata_gt[:point_num,2],posedata_gt[:point_num,0],-posedata_gt[:point_num,1],c='r',s =0.5)
plt.show()
