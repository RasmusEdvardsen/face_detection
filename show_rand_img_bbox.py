from PIL import Image, ImageDraw
import pandas as pd
from random import randrange

df = pd.read_csv('annotations\\transformed_normalized\\annotations_transformed_normalized.csv')

annotation = df.iloc[randrange(df.shape[0])]
name = annotation['id'].split('.')[0] + '.png'

img = Image.open(f'images\\normalized\\{name}')

img_draw = ImageDraw.Draw(img)

bbox = (annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax'])
img_draw.rectangle(bbox)

img.show()