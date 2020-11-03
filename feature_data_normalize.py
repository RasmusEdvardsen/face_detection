# resize image to some size, then resize bboxes with same ratio

# width = xxx
# height = xxx

# for each image
#   if annotation exists
#       get width, height ratio for resize
#       resize images
#       round(num, decimals) annotations by ratio
#
#       save annotations to annotations/normalized_transformed
#       save images to images/normalized