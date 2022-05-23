from tf.transformations import quaternion_from_matrix, quaternion_matrix
from geometry_msgs.msg import Point, Quaternion, Transform, Pose
import numpy as np

'''High-level functions'''

def BuildMatrix(translation, quaternion):
    tfmatrix = quaternion_matrix(quaternion)
    tfmatrix[0][3] = translation[0]
    tfmatrix[1][3] = translation[1]
    tfmatrix[2][3] = translation[2]
    return tfmatrix

def InvertTransform(translation, quaternion):
    tfmatrix = BuildMatrix(translation, quaternion)
    xirtamft = np.linalg.inv(tfmatrix)
    inverted = TransformFromMatrix(xirtamft)
    return inverted

def TransformFromMatrix(old_matrix):
    [translation, quaternion] = ExtractFromMatrix(old_matrix)
    transformed = TransformFromComponents(translation, quaternion)
    return transformed

def ExtractFromMatrix(old_matrix):
    quaternion = quaternion_from_matrix(old_matrix)
    translation = [old_matrix[0][3], old_matrix[1][3], old_matrix[2][3]]
    return [translation, quaternion]

def TransformToMatrix(old_transform):
    [translation, quaternion] = ComponentsFromTransform(old_transform)
    tfmatrix = BuildMatrix(translation, quaternion)
    return tfmatrix


def ComponentsFromTransform(old_transform):
    translation = [old_transform.translation.x, old_transform.translation.y, old_transform.translation.z]
    quaternion = [old_transform.rotation.x, old_transform.rotation.y, old_transform.rotation.z,
                  old_transform.rotation.w]
    return [translation, quaternion]


def ComposeTransforms(transform1, transform2):
    tfmatrix1 = TransformToMatrix(transform1)
    tfmatrix2 = TransformToMatrix(transform2)
    composedmatrix = np.dot(tfmatrix1, tfmatrix2)
    composed = TransformFromMatrix(composedmatrix)
    return composed

def ComposePoses(pose1, pose2):
    transform1 = PoseToTransform(pose1)
    transform2 = PoseToTransform(pose2)
    composed = PoseFromTransform(ComposeTransforms(transform1, transform2))
    return composed




'''Conversion functions'''

def PoseToTransform(old_pose):
    transformed = Transform()
    transformed.translation.x = old_pose.position.x
    transformed.translation.y = old_pose.position.y
    transformed.translation.z = old_pose.position.z
    transformed.rotation.x = old_pose.orientation.x
    transformed.rotation.y = old_pose.orientation.y
    transformed.rotation.z = old_pose.orientation.z
    transformed.rotation.w = old_pose.orientation.w
    return transformed

def PoseFromTransform(old_transform):
    posed = Pose()
    posed.position.x = old_transform.translation.x
    posed.position.y = old_transform.translation.y
    posed.position.z = old_transform.translation.z
    posed.orientation.x = old_transform.rotation.x
    posed.orientation.y = old_transform.rotation.y
    posed.orientation.z = old_transform.rotation.z
    posed.orientation.w = old_transform.rotation.w
    return posed


def TransformFromComponents(translation, quaternion):
    transformed = Transform()
    transformed.translation.x = translation[0]
    transformed.translation.y = translation[1]
    transformed.translation.z = translation[2]
    transformed.rotation.x = quaternion[0]
    transformed.rotation.y = quaternion[1]
    transformed.rotation.z = quaternion[2]
    transformed.rotation.w = quaternion[3]
    return transformed
