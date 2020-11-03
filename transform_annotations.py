"""
Object detection is non trivial.
If there is object plurality in images, it adds further complexity.
For now, only use images with one face.
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import xml.etree.ElementTree as etree

columns = ['id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
df = pd.DataFrame(columns=columns)
df = df.set_index('id')

path = 'annotations'
files = [f for f in listdir(path) if isfile(join(path, f))]

for annotations in files:
    tree = etree.parse(join(path, annotations))
    node = tree.getroot()
    faces = node.findall('object')
    if len(faces) == 1:
        annotation = faces[0]
        bbox = annotation.find('bndbox')
        # might need column specification
        data_row = [annotations, 1, bbox.find('xmin').text, bbox.find('ymin').text, bbox.find('xmax').text, bbox.find('ymax').text]
        a_row = pd.DataFrame([data_row], columns=columns)
        a_row = a_row.set_index('id')
        df = pd.concat([df, a_row])
        print(df)
df.to_csv('annotations_transformed.csv')