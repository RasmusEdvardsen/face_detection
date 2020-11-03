from PIL import Image, ImageDraw
import pandas as pd

width_height = (350, 450)
path = 'images\\raw\\'

df = pd.read_csv('annotations\\transformed\\annotations_transformed.csv')

# TODO 300x300 or less
# TODO look at image general sizes

for index, row in df.iterrows():
    name = row['id'].split('.')[0] + '.png'

    img = Image.open(path + name)
    img_w_ratio = img.width / 350
    img_h_ratio = img.height / 450
    img = img.resize(width_height)

    df.at[index, 'xmin'] = round(row['xmin']/img_w_ratio)
    df.at[index, 'ymin'] = round(row['ymin']/img_h_ratio)
    df.at[index, 'xmax'] = round(row['xmax']/img_w_ratio)
    df.at[index, 'ymax'] = round(row['ymax']/img_h_ratio)

    img.save(f'images/normalized/{name}')

df = df.set_index('id')
df.to_csv('annotations\\transformed_normalized\\annotations_transformed_normalized.csv')