import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

def analyze_labels(xmlInputFiles):
    print("Analyzing labels...")
    print("# xml files: {}".format(len(xmlInputFiles)))


def relabel_label(xmlInputFiles, xmlOutputPath, oldLabel, newLabel):
    print("Relabeling labels...")
    i = 0 # iterator per file
    n = 0 # iterator of change objects
    for x in xmlInputFiles:
        i += 1
        print("Progress {:2.1%}: {}".format(i / len(xmlInputFiles), x.name), end="\x1b[1K\r")
        tree = ET.parse(x)
        root = tree.getroot()
        for className in root.iter('name'):
            if className.text == oldLabel:
                className.text = newLabel
                n += 1
        tree.write(x)
    print("\n")
    print("Relabeling completed! {} objects relabeled".format(n))
        

def delete_label(xmlInputFiles, xmlOutputPath, labelName):
    print("Delete labels...")
    i = 0 # iterator per file
    n = 0 # iterator of change objects
    for x in xmlInputFiles:
        i += 1
        print("Progress {:2.1%}: {}".format(i / len(xmlInputFiles), x.name), end="\x1b[1K\r")
        tree = ET.parse(x)
        root = tree.getroot()
        for object in root.findall('object'):
            className = object.find('name')
            if className.text == labelName:
                root.remove(object)
                n += 1
        tree.write(x)
    print("\n")
    print("Deleting completed! {} objects deleted".format(n))

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Modify Pascal VOC annotations"
    )
    parser.add_argument("xml_dir", help="Directory path of the xml files.", type=str)
    return parser.parse_args()

def main(args):
    print("Pascal VOC (xml) annotation modifier by Daniel Ahlers")
    
    xmlInputPath = Path(args.xml_dir).glob('**/*.xml') # get all xml files inside 'xml_dir'
    xmlInputFiles = [x for x in xmlInputPath if x.is_file()] # create a list of all xml files
    print("Number of xml files in xml_dir found: {nXmlFiles}".format(nXmlFiles = len(xmlInputFiles)))
    xmlOutputPath = xmlInputPath

    while True:
        print("Please choose an option:\n1: list all labels and count them\n2: relabel one class\n3: delete all labels of a specific class\nQ: to quit")
        userInput = input()
        if userInput == 'Q' or userInput == 'q': # check if the user wants to quit the program
            break
        elif userInput == '':
            print("Invalid input")
        else:
            userInput = int(userInput) # typecast userInput to int
            
        if userInput == 1:
            analyze_labels(xmlInputFiles)

        elif userInput == 2:
            while True:
                oldLabel = input("Enter the old label or leave empty to go back: ")
                if oldLabel == '':
                    print("Input is empty. Abort and go back")
                    break
                newLabel = input("Enter the new label: ")

                safeguard = input('Are your sure, you want to relabel: "{oldLabelName}" -> "{newLabelName}"? y/n: '.format(oldLabelName = oldLabel, newLabelName = newLabel))
                if safeguard.lower() == 'y' or safeguard == 'yes':
                    relabel_label(xmlInputFiles, xmlOutputPath, oldLabel, newLabel)
                elif safeguard.lower() == 'n' or safeguard == 'no':
                    print("Abort relabeling")
                    break
                else:
                    print("Invalid input")

        elif userInput == 3:
            while True:
                deleteLabel = input("Enter the label you want to delete all instances of, leave empty to go back: ")
                if deleteLabel == '':
                    break
                safeguard = input('Are your sure, you want to delete all labels with the name: "{labelName}"? y/n: '.format(labelName = deleteLabel))
                if safeguard.lower() == 'y' or safeguard == 'yes':
                    delete_label(xmlInputFiles, xmlOutputPath, deleteLabel)
                elif safeguard.lower() == 'n' or safeguard == 'no':
                    print("Abort deleting")
                    pass
                else:
                    print("Invalid input")
                
        else:
            print("Invalid input")

if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)