import glob


# list all .pdf in the directory
files = glob.glob('knowledge_base/*.pdf')
print(files)
for file in files:
    # read pdf and populate the df
    with open(file, 'rb') as file:
        print('ok')
