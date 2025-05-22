import os
import scipy.io as sio
import numpy as np

data_dir = "data/SUN/"



data = sio.loadmat(os.path.join(data_dir,'SUNAttributeDB','images.mat'))
image_names = data['images']
print(image_names.shape,image_names[0][0][0])

with open(os.path.join(data_dir,'SUNAttributeDB','img_paths.txt'), 'w') as f:
    for p in image_names:
        f.write(p[0][0]+'\n')

class_names = []
for path in data['images']:
    #print(path[0][0])
    classname = path[0][0].split("/sun_")[0]
    if classname not in class_names:
        class_names.append(classname)
print(len(class_names),class_names[0])



data = sio.loadmat(os.path.join(data_dir,'SUNAttributeDB','attributes.mat'))
attr_names = [x[0][0] for x in data['attributes']]
print(len(attr_names),attr_names[:3])



data = sio.loadmat(os.path.join(data_dir,'SUNAttributeDB','attributeLabels_continuous.mat'))
attr_labels = data['labels_cv']
print(attr_labels.shape)


"""
1 not visible
2 guessing
3 probably
4 definitely
"""


print("total imgs: ", len(image_names))

final_text = []
for img_id,p in enumerate(image_names):
    classname = path[0][0].split("/sun_")[0]
    #class_id = class_names.index(classname)

    full_sentense = "Image id %s belong to class %s." % (img_id, classname.split("/")[-1])
    attr_sentense = ""

    attr_items = attr_labels[img_id]

    for attr_id,attr_item in enumerate(attr_items):
        if attr_item<0.3:
            continue
        elif attr_item<0.6:
            cert_word = "is guessing"
        elif attr_item<0.9:
            cert_word = "is probably"
        else:
            cert_word = "is definitely"

        attr_value = attr_names[attr_id]

        attr_text = "It %s %s." % (cert_word,attr_value)

        attr_sentense+=attr_text

    final_text.append(full_sentense +" "+attr_sentense)

with open(data_dir+"/sun_attr_text.txt", 'w') as f:
    for text in final_text:
        f.write(text+"\n")


