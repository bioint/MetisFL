import os

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import ndimage


NEW_CSV_OUTPUT_PATH = "./train_8_distorted.csv"
DISTORTED_SCANS_OUTPUT_DIR = "/lfs1/stripeli/neuroimaging_data/ukbb"
SCAN_VOL_COL = "9dof_2mm_vol"


def plot_slices(scan, fout='outfile.png'):
    fig = plt.figure()
    fig=plt.figure(figsize=(9, 3))
    fig.add_subplot(1,3,1)
    plt.grid()
    plt.imshow(scan[45,:,:])
    fig.add_subplot(1,3,2)
    plt.grid()
    plt.imshow(scan[:,54,:])
    fig.add_subplot(1,3,3)
    plt.grid()
    plt.imshow(scan[:,:,45])
    fig.savefig(fout, bbox_inches = 'tight')


def distort_images(csv_file, blurring_image=False, noisy_image=False, sigma=0.3):

    assert blurring_image is True or noisy_image is True

    data = pd.read_csv(csv_file)
    path_saves = []
    for i in range(len(data)):
        print("IMAGE:", i+1)
        scan_path = data.iloc[i][SCAN_VOL_COL]
        subj_id, fname = scan_path.rsplit("/", 2)[1:3]
        scan = nib.load(scan_path)
        affine = scan.affine
        scan = scan.get_fdata()

        distorted_image = None
        if blurring_image:
            scan_blurred = ndimage.gaussian_filter(scan, sigma, order=0, mode="reflect")
            distorted_image = scan_blurred
            output_path = DISTORTED_SCANS_OUTPUT_DIR + "/9DOF_blurred_scans/{}_blurred_sigma{}/"

        elif noisy_image:
            gaussian_noise = np.random.normal(0, sigma, size=scan.shape)
            scan_noisy = scan + gaussian_noise
            distorted_image = scan_noisy
            output_path = DISTORTED_SCANS_OUTPUT_DIR + "/9DOF_noisy_scans/{}/noisy_sigma{}/"

        str_sigma = ''.join(str(sigma).split('.'))
        corrupted_subj_path = output_path.format(subj_id, str_sigma)
        if not os.path.isdir(corrupted_subj_path):
            os.makedirs(corrupted_subj_path)

        distorted_image = nib.Nifti1Image(distorted_image, affine)
        scan_path = corrupted_subj_path + fname
        distorted_image.to_filename(scan_path)
        path_saves.append(scan_path)

    data[SCAN_VOL_COL] = path_saves
    data.to_csv(NEW_CSV_OUTPUT_PATH, index=False)


def testing_distortion():
    SCAN_PATH = "nu_T1_brain_2mm.nii.gz"
    scan = nib.load(SCAN_PATH)
    affine = scan.affine
    scan = scan.get_fdata()
    plot_slices(scan, fout="image_original.png")

    scan_blurred = ndimage.gaussian_filter(scan, 0.3, order=0, mode="reflect")
    plot_slices(scan_blurred, fout="image_blurred.png")

    gaussian_noise = np.random.normal(0, 3, size=scan.shape)
    scan_noisy = scan + gaussian_noise
    plot_slices(scan_noisy, fout="image_noisy.png")


if __name__=="__main__":
    csv_input_file = "../uniform_datasize_iid_x8clients/without_validation/train_8.csv"
    distort_images(csv_input_file, noisy_image=True, sigma=3)

