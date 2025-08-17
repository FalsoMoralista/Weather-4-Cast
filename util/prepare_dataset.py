import json
import os
import shutil

f = open('/home/rtcalumby/adam/luciano/iNaturalist2019/val2019.json')
validation_data = json.load(f)

for image in validation_data['images']:
    image = image['file_name'].split('/')
    parent_category = image[1]
    directory_id = image[2]
    image_path = image[3]

    if not os.path.exists('/home/rtcalumby/adam/luciano/iNaturalist2019/val/' + directory_id):
      os.makedirs('/home/rtcalumby/adam/luciano/iNaturalist2019/val/' + directory_id, exist_ok=True)

    src = '/home/rtcalumby/adam/luciano/iNaturalist2019/train_val2019/' + parent_category +'/'+ directory_id +'/' + image_path
    destination = '/home/rtcalumby/adam/luciano/iNaturalist2019/val/' + directory_id
    #shutil.move(src, destination)
    