# Read patterns from remove.txt
with open('list_sync_test.txt', 'r') as remove_file:
    patterns = [line.strip() for line in remove_file]

# Read lines from list.txt
with open('list_sync_all.txt', 'r') as list_file:
    lines = list_file.readlines()

# Write to adjusted_file.txt excluding lines that match any pattern
with open('list_syncs_train.txt', 'w') as adjusted_file:
    for line in lines:
        if line.strip() not in patterns:
            adjusted_file.write(line)
