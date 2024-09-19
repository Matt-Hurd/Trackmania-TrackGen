import json

def extract_block_events(file_path):
    # List to store extracted data
    extracted_data = []

    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate through each dictionary in the array
    for item in data:
        # Check if 'event_type' key exists and matches the specified types
        if 'event_type' in item and item['event_type'] in ['BLOCK_ENTER', 'BLOCK_EXIT']:
            # Extract the required keys
            extracted_item = {
                key: item.get(key, None) for key in [
                    'BlockDirection', 'BlockName', 'BlockPageName',
                    'BlockMaterialName', 'BlockPosition'
                ]
            }
            extracted_data.append(extracted_item)

    return extracted_data

# Example usage
file_path = 'predictions.json'
result = extract_block_events(file_path)

# Print the result
print(json.dumps(result, indent=2))