import contour2ucm
import numpy as np

pb_ori = np.random.uniform(size=(8, 128, 128)).astype(np.float32)
pb = np.random.uniform(size=(128, 128)).astype(np.float32)
print pb


contour2ucm.contour2ucm_py(pb, pb_ori)

