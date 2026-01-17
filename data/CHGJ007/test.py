import  nibabel as nb

file_path = '/home/reza/Documents/Arezoo_Codes/2dto3d/ai4elife-main/src/output/preprocessed/hector_default_3d_dir/CHGJ007/pet.nii'
img = nb.load(file_path)

data = img.get_fdata()
print(data.shape)
