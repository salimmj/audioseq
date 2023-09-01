import os
import xml.etree.ElementTree as ET

def get_unique_events(folder_path):
    unique_events = set()
    
    # Loop over all files in the specified directory
    for filename in os.listdir(folder_path):
        
        # Check if the file is an XML file
        if filename.endswith('.xml'):
            
            # Construct full file path
            full_path = os.path.join(folder_path, filename)
            
            # Parse the XML file
            tree = ET.parse(full_path)
            root = tree.getroot()
            
            # Iterate through each 'item' in 'events' section in XML
            for item in root.find('events'):
                class_id = item.find('CLASS_ID').text
                class_name = item.find('CLASS_NAME').text
                
                # Add unique (CLASS_ID, CLASS_NAME) tuples to the set
                unique_events.add((class_id, class_name))
                
    return unique_events

# Folder path containing XML files
folder_path = 'training'

# Get the unique set of all event (class id, class name) tuples
unique_events = get_unique_events(folder_path)

# Print or further process the unique events
print(unique_events)
