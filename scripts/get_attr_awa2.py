import os


data_dir = "data/AwA2"

image_class_labels = data_dir+"/AwA2-labels.txt"
with open(image_class_labels, 'r') as f:
    image_class_labels_lines =  f.readlines()

image_names = data_dir+"/AwA2-filenames.txt"
with open(image_names, 'r') as f:
    image_names_lines =  f.readlines()

label_id2name = {}
label_name2id = {}
classes = data_dir+"/Animals_with_Attributes2/classes.txt"
with open(classes, 'r') as f:
    classes_lines =  f.readlines()
    for line in classes_lines:

        class_id, class_name = line.strip().split("\t")
        class_id = int(class_id)-1
        label_id2name[class_id] = class_name
        label_name2id[class_name] = class_id

attr_id2name = {}
attrs = data_dir+"/Animals_with_Attributes2/predicates.txt"
with open(attrs, 'r') as f:
    attrs_lines =  f.readlines()
    for line in attrs_lines:
        attr_id, attr_name = line.strip().split("\t")
        attr_id = int(attr_id)-1
        attr_id2name[attr_id] = attr_name

class2attr = {}
attrs = data_dir+"/Animals_with_Attributes2/predicate-matrix-continuous.txt"
with open(attrs, 'r') as f:
    attrs_lines =  f.readlines()
    for cid,line in enumerate(attrs_lines):
        #print(line.strip().replace("   "," ").split(" "))
        attr_v = [float(x) for x in line.strip().replace("   "," ").replace("  "," ").split(" ")]
        class2attr[cid] = attr_v


"""
1 not visible
2 guessing
3 probably
4 definitely
"""


print("total imgs: ", len(image_names_lines))

final_text = []
for img_id,line in enumerate(image_names_lines):
    class_name = line.strip().split("_")[0]
    #print(line)
    class_id = label_name2id[class_name]
    
    full_sentense = "Image id %s belong to class %s." % (img_id, class_name)
    attr_sentense = ""

    attr_items = class2attr[class_id]

    for attr_id,attr_item in enumerate(attr_items):
        if attr_item<20:
            continue
        elif attr_item<50:
            cert_word = "is guessing"
        elif attr_item<90:
            cert_word = "is probably"
        else:
            cert_word = "is definitely"

        #attr_id, is_present, certainty_id = attr_item

        attr_value = attr_id2name[attr_id]

        attr_text = "It %s %s." % (cert_word,attr_value)
        #print(attr_text)
        attr_sentense+=attr_text

    print(full_sentense, attr_sentense)
    final_text.append(full_sentense +" "+attr_sentense)
    #break

with open(data_dir+"/awa2_attr_text.txt", 'w') as f:
    for text in final_text:
        f.write(text+"\n")


