# import os
# import pandas as pd

# # Read the excel file containing the filenames
# filename_df = pd.read_excel('filename_list.xlsx')

# # Convert the filename column to a set for efficient membership testing
# filename_set = set(filename_df['filename'])

# # Loop through all files in the folder
# for filename in os.listdir():
#     # Check if the file is not in the list of filenames
#     if filename not in filename_set:
#         # If not, delete the file
#         os.remove(filename)