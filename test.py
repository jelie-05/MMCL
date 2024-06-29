import yaml

# Load YAML configuration
with open(r'C:\Users\jerem\OneDrive\Me\StudiumMaster\00_Semesterarbeit\Project\MMSiamese\configs\configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Retrieve learning rate and convert to float
learning_rate = float(config['train']['lr'])

# Now learning_rate is a float
print(type(learning_rate))  # Output: 0.001
