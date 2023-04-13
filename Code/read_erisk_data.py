import xml.etree.ElementTree as ET

#opens an xml subject file, and converts it into a list, where each entry is a dictionary.
#Each dictionary has as keys Subject, Title, Text and Date. SOMETIMES TITLE AND TEXT MIXED UP?
def read_subject_writings(subject_file):
    writings = []
    with open(subject_file) as file:
        contents = file.read()
        root = ET.fromstring(contents)

        try:
            subject = root.findall('ID')[0].text.strip()
        except Exception:
            print('Cannot extract ID', contents[:500], '\n-------\n')
        for w in root.iter('WRITING'):
            subject_writings = {'subject': subject}
            for title in w.findall('TITLE'):
                subject_writings['title'] = title.text
            for text in w.findall('TEXT'):
                subject_writings['text'] = text.text
            for date in w.findall('DATE'):
                subject_writings['date'] = date.text
            writings.append(subject_writings)

    return writings