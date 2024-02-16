import re

# log_file_path = '/home/hinton/uemerson/OASIS/logs/vxm_2_max_mse_1_diffusion_1/logfile.log'
# log_file_path = '/home/hinton/uemerson/OASIS/logs/vxm_2_bpca_mse_1_diffusion_1/logfile.log'
# log_file_path = '/home/hinton/uemerson/OASIS/logs/vxm_2_mse_1_diffusion_1/logfile.log'
# log_file_path = '/home/hinton/uemerson/OASIS/logs/vxm_2_bpca_revert_mse_1_diffusion_1/logfile.log'
# log_file_path = '/home/hinton/uemerson/OASIS/logs/vxm_2_max_bpca_mse_1_diffusion_1/logfile.log'
log_file_path = '/home/hinton/uemerson/OASIS/logs/vxm_2_bpca_max_mse_1_diffusion_1/logfile.log'

pattern = re.compile(r'time taken: (\d+\.\d+) seconds')

seconds = []

with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            seconds.append(float(match.group(1)))

total_seconds = sum(seconds)

# Convert seconds to days, hours, minutes, and remaining seconds
days = int(total_seconds // (24 * 3600))
hours = int((total_seconds % (24 * 3600)) // 3600)
minutes = int((total_seconds % 3600) // 60)
remaining_seconds = total_seconds % 60

# Print the result
print(f"total seconds: {total_seconds}")
print(
    f'Total time: {days} days, {hours} hours, {minutes} minutes, and {remaining_seconds:.6f} seconds.')
