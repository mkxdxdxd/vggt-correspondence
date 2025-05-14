import torch
from safetensors.torch import load_file

# Load the safetensors file into a state dictionary
state_dict = load_file("/media/multiview-gen/genwarp/exp_output/vanilla_unified_mark_three_depth_realestate_250220_032051/checkpoint-36000/model.safetensors")

# Extract keys starting with "geometry_unet." and remove the prefix from each key
geo = {key[len("geometry_unet."):]: value 
       for key, value in state_dict.items() 
       if key.startswith("geometry_unet.")}

# Save the updated dictionary to a .pth file
torch.save(geo, "/media/multiview-gen/checkpoints/unified_depth/geometry_unet.pth")


den = {key[len("denoising_unet."):]: value 
       for key, value in state_dict.items() 
       if key.startswith("denoising_unet.")}

# Save the updated dictionary to a .pth file
torch.save(den, "/media/multiview-gen/checkpoints/unified_depth/denoising_unet.pth")


ref = {key[len("reference_unet."):]: value 
       for key, value in state_dict.items() 
       if key.startswith("reference_unet.")}

# Save the updated dictionary to a .pth file
torch.save(ref, "/media/multiview-gen/checkpoints/unified_depth/reference_unet.pth")


pose = {key[len("pose_guider."):]: value 
       for key, value in state_dict.items() 
       if key.startswith("pose_guider.")}

# Save the updated dictionary to a .pth file
torch.save(pose, "/media/multiview-gen/checkpoints/unified_depth/pose_guider.pth")