import numpy as np
np.savez(target_forward=target_forward.numpy(force=True), target_backward=target_backward.numpy(force=True),
         pred_forward=pred_forward.numpy(force=True), pred_backward=pred_backward.numpy(force=True), file="./rescue.npz")
