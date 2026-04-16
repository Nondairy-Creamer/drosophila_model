import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from FlyEye.utils import sph_from_cart, flip_horizontal, rotate_minus_60, rotate_60
ROOT = Path.cwd()  

# script to extract fly view data for use in mapping ommatidia 
DATA_DIR = ROOT / "data"
fly_data = DATA_DIR / "fly"
# shift to align left and right eyes in pleasing layout
P_SHIFT = -18
Q_SHIFT = -18
plot = True

# load data for right eye
df = pd.read_parquet(os.path.join(fly_data,'column_info_df.parquet'))

# unpack
right_x,right_y,right_z = np.array(df['x']),np.array(df['y']),np.array(df['z'])
right_p,right_q = np.array(df['p']),-np.array(df['q'])

# convert to visual degrees
right_azimuth,right_elevation = sph_from_cart(right_x,right_y,right_z)

# create data for left eye (flip sign of y)
left_x, left_y, left_z = right_x, -right_y, right_z

left_azimuth,left_elevation = sph_from_cart(left_x,left_y,left_z)


inter_p,inter_q = rotate_minus_60(right_p, right_q)
left_p,left_q = flip_horizontal(inter_p,inter_q)

left_p += P_SHIFT
left_q += Q_SHIFT
if plot:
    # visualize right eye
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(right_x,right_y,right_z,c=right_azimuth)
    plt.show()
    # visualize left eye
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(left_x,left_y,left_z,c=left_azimuth)
    plt.show()


# save data for right eye
np.savez(os.path.join(fly_data,'right_eye.npz'),azimuth=right_azimuth,elevation=right_elevation,p=right_p,q=right_q,x=right_x,y=right_y,z=right_z)
np.savez(os.path.join(fly_data,'left_eye.npz'),azimuth=left_azimuth,elevation=left_elevation,p=left_p,q=left_q,x=left_x,y=left_y,z=left_z)


#  combine for both eyes
both_x = np.concatenate((left_x,right_x))
both_y = np.concatenate((left_y,right_y))
both_z = np.concatenate((left_z,right_z))

both_azimuth = np.concatenate((left_azimuth,right_azimuth))
both_elevation = np.concatenate((left_elevation,right_elevation))

both_p = np.concatenate((left_p,right_p))
both_q = np.concatenate((left_q,right_q))

both_eye_indicator = np.concatenate((np.zeros(len(left_x)),np.ones(len(right_x))))
# find pleasing layout for second eye 
# visualize 
if plot:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(both_x,both_y,both_z,c=both_azimuth)
    plt.show()

# save data for both eyes
np.savez(os.path.join(fly_data,'both_eyes.npz'),azimuth=both_azimuth,elevation=both_elevation,p=both_p,q=both_q,x=both_x,y=both_y,z=both_z,eye_id=both_eye_indicator)

