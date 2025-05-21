import os  



read_dir = "data/xlsa17/data"

data_dirs = ['AWA2','CUB','SUN']

for data_dir in data_dirs:
    total_lines = []
    with open(os.path.join(read_dir, data_dir, 'trainvalclasses.txt'),'r') as f:
        lines = f.readlines()
        total_lines.extend(lines)
    with open(os.path.join(read_dir, data_dir, 'testclasses.txt'),'r') as f:
        lines = f.readlines()
        total_lines.extend(lines)

    total_lines.sort()
    with open(os.path.join(read_dir, data_dir, 'allclasses.txt'),'w') as f:
        for line in total_lines:
            f.write(line)