import os


data_dir = "data/CUB_200_2011"

image_class_labels = data_dir+"/image_class_labels.txt"
with open(image_class_labels, 'r') as f:
    image_class_labels_lines =  f.readlines()

label_id2name = {}
classes = data_dir+"/classes.txt"
with open(classes, 'r') as f:
    classes_lines =  f.readlines()
    for line in classes_lines:
        class_id, class_name = line.strip().split(" ")
        label_id2name[class_id] = class_name

# <image_id> <attribute_id> <is_present> <certainty_id> <time>
#<is_present> is 0 or 1 (1 denotes that the attribute is present)
image_attribute_labels = data_dir+"/attributes/image_attribute_labels.txt"
imgid2attr = {}
with open(image_attribute_labels, 'r') as f:
    image_attribute_labels_lines =  f.readlines()
    for line in image_attribute_labels_lines:
        # print(line)
        img_id, attr_id, is_present, certainty_id = line.strip().split(" ")[:4]

        if attr_id=='1':
            imgid2attr[img_id] = [[attr_id, is_present, certainty_id]]
        else:
            imgid2attr[img_id] += [[attr_id, is_present, certainty_id]]


attr_id2text = {}
attributes = data_dir+"/attributes.txt"
with open(attributes, 'r') as f:
    attributes_lines =  f.readlines()
    for line in attributes_lines:
        attr_id, attr_text = line.strip().split(" ")
        attr_id2text[attr_id] = attr_text

"""
1 not visible
2 guessing
3 probably
4 definitely
"""
cert_id2name = {}
certainties = data_dir+"/attributes/certainties.txt"
with open(image_attribute_labels, 'r') as f:
    certainties_lines =  f.readlines()
    for line in certainties_lines:
        items = line.strip().split(" ")
        cert_id = items[0]
        cert_text = " ".join(items[1:])
        cert_id2name[cert_id] = cert_text 


print("total imgs: ", len(image_class_labels_lines))

final_text = []
for line in image_class_labels_lines:
    img_id, class_id = line.strip().split(" ")
    
    class_name = label_id2name[class_id]
    #print(img_id, class_id, class_name)
    attr_items = imgid2attr[img_id]
    #print(len(attr_items), attr_items[:2])
    
    full_sentense = "Image id %s belong to class %s." % (img_id, class_name)
    attr_sentense = ""

    for attr_item in attr_items:
        attr_id, is_present, certainty_id = attr_item
        if is_present=='1' and int(certainty_id)>1:
            #print(attr_id, certainty_id, attr_id2text[attr_id])
            attr_items = attr_id2text[attr_id].strip().split("::")
            attr_desc = " ".join(attr_items[0].split("_")[1:])
            attr_value = " ".join(attr_items[1].split("_"))

            cert_word = "is definitely"
            if certainty_id in ['2','3']:
                cert_word = "is probably"

            attr_text = "It's %s %s %s." % (attr_desc,cert_word,attr_value)
            #print(attr_text)
            attr_sentense+=attr_text

    print(full_sentense, attr_sentense)
    final_text.append(full_sentense +" "+attr_sentense)
    #break

with open(data_dir+"/cub_attr_text.txt", 'w') as f:
    for text in final_text:
        f.write(text+"\n")


