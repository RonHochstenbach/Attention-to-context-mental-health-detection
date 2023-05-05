import os
import xml.etree.ElementTree as ET

# directory containing XML files
xml_dir = "path/to/xml/files"

# create an Element object to hold the merged XML content
merged_xml = ET.Element('root')

# loop through all XML files in the directory
for filename in os.listdir(xml_dir):
    if filename.endswith('.xml'):
        # parse the XML file and get the root element
        xml_path = os.path.join(xml_dir, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # append the root element to the merged XML
        merged_xml.append(root)

# create a new ElementTree object with the merged XML
merged_tree = ET.ElementTree(merged_xml)

# write the merged XML to a file
merged_tree.write('merged.xml')