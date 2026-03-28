from PIL import Image
import numpy as np

mask = np.array(Image.open(r'DentAI.v2i.coco-segmentation\train\masks_multiclass\4_jpeg.rf.12479d7eeef261cce943d91fbb789a0d.png'))

# Scale up for viewing
visual = (mask * 80).astype(np.uint8)  # multiply so eyes can see
Image.fromarray(visual).save('visual_check.png')
print("Saved! Open visual_check.png to see your mask")