import numpy as np
import SimpleITK as sitk

def calc_centroid(label, labeled_array):
    indices = np.array(np.where(labeled_array == label))
    centroid = np.mean(indices, axis=1).astype(int)
    return centroid

def extract_position_vector(data, label):

    image = sitk.GetImageFromArray(data)
    # Convert the labeled region to a binary mask
    binary_mask=image
    # Create the label shape statistics filter
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.ComputeOrientedBoundingBoxOn ()
    lsif.Execute(binary_mask)

    orientation_matrix = np.array(lsif.GetOrientedBoundingBoxDirection(label)).reshape((3, 3))
    # Extract rotation angles in radians
    theta_x = np.arctan2(orientation_matrix[2, 1], orientation_matrix[2, 2])
    theta_y = np.arctan2(-orientation_matrix[2, 0], np.sqrt(orientation_matrix[2, 1]**2 + orientation_matrix[2, 2]**2))
    theta_z = np.arctan2(orientation_matrix[1, 0], orientation_matrix[0, 0])

    centroid = np.array(lsif.GetCentroid(label))[::-1]
    size_d = np.array(lsif.GetOrientedBoundingBoxSize(label))[::-1]
    thetas = np.array([theta_x, theta_y, theta_z])
    return centroid,size_d,thetas


def resample_image(image, new_spacing, new_direction,new_origin=None,interp_method=sitk.sitkLinear):
    # Get original properties
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()

    # Compute new size based on original physical dimensions
    original_physical_size = [sz * spc for sz, spc in zip(original_size, original_spacing)]
    new_size = [int(round(phys_sz / spc)) for phys_sz, spc in zip(original_physical_size, new_spacing)]

    # Define the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)  
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interp_method)

    # Resample the image
    resampled_image = resampler.Execute(image)
   
    return resampled_image
