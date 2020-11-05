from PIL import Image, ImageDraw
import pandas as pd

width_height = (32, 32)

# todo use join the file path + file names.

def normalize(file_raw, file_norm, img_raw, img_norm):
    """
    Resizes images, divides features by resize ratio, saves features, images
    """

    df = pd.read_csv(file_raw)

    for index, row in df.iterrows():
        name = row['id'].split('.')[0] + '.png'

        img = Image.open(img_raw + name)
        img_w_ratio = img.width / 350
        img_h_ratio = img.height / 450
        img = img.resize(width_height)

        df.at[index, 'xmin'] = round(row['xmin']/img_w_ratio)
        df.at[index, 'ymin'] = round(row['ymin']/img_h_ratio)
        df.at[index, 'xmax'] = round(row['xmax']/img_w_ratio)
        df.at[index, 'ymax'] = round(row['ymax']/img_h_ratio)

        img.save(f'{img_norm}\\{name}')

    df = df.set_index('id')
    df.to_csv(file_norm)

normalize('annotations\\makeml\\transformed\\annotations_transformed.csv', 
    'annotations\\makeml\\transformed_normalized\\annotations_transformed_normalized.csv',
    'images\\makeml\\raw\\',
    'images\\makeml\\normalized\\')