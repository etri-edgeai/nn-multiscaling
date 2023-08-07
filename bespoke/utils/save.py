import os
import json

import tensorflow as tf

def student_model_save(model, dir_, inplace=False, prefix=None, postfix=""):

    if prefix is None:
        prefix = ""

    if inplace:
        student_house_path = dir_
        assert os.path.exists(student_house_path)
    else:
        student_house_path = os.path.join(dir_, "students") 
        if not os.path.exists(student_house_path):
            os.mkdir(student_house_path)

    cnt = 0
    basename = prefix+"student"
    if postfix != "":
        filepath = os.path.join(student_house_path, "%s%s.h5" % (basename, postfix))
    else:
        filepath = os.path.join(student_house_path, "%s_%d.h5" % (basename, cnt))
        while os.path.exists(filepath):
            cnt += 1
            filepath = os.path.join(student_house_path, "%s_%d.h5" % (basename, cnt))

    tf.keras.models.save_model(model, filepath, overwrite=True, include_optimizer=False)
    print("model saving... %s done" % filepath)
    return filepath

def running_time_dump(model_filepath, running_time):
    student_dir = os.path.dirname(model_filepath)
    basename = os.path.splitext(os.path.basename(model_filepath))[0]
    with open(os.path.join(student_dir, "%s.log" % basename), "w") as file_:
        json.dump(running_time, file_)
