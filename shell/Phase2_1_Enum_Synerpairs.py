import os
import csv
import ast

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))

# Define input and output paths
input_path = os.path.join(project_root, 'output', 'Phase1_Filtered_SynerPairLists.csv')
output_path = os.path.join(project_root, 'output', 'Phase2_Enumerated_pairs.csv')

# Open the original CSV file
with open(input_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    # Open the new CSV file for writing enumerated results
    with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Write the header
        writer.writerow(['index', 'synerpair'])
        
        # Enumerate all list values in the rows and write to the new file
        index_counter = 0
        seen_elements = set()
        for row in reader:
            # Get the value from the 'Synerpairlist' column and parse it as a list
            value_str = row['Synerpairlist']
            value_list = ast.literal_eval(value_str)
            
            # Enumerate the values in the list and skip duplicates
            for element in value_list:
                if element not in seen_elements:
                    writer.writerow([index_counter, element])
                    seen_elements.add(element)
                    index_counter += 1

print("Enumeration completed and saved to Phase2_Enumerated_pairs.csv")
