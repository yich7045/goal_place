# mesh2img_projection

## Function
Projection of a mesh in world frame onto a 2D image.

## Requirement
1. Camera Info (Intrinsics)
2. Transformation between Camera and the world frame (Extrinsics)
3. 2D image 
4. Mesh -- a thing you want to project


## How to use
1. Update config/realsense_param.json
2. Update TODOs in pcd2img_projection.py
3. ```$ python pcd2img_projection.py  ```

Result will be saved in 'result/' folder.
