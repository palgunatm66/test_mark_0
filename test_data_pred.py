import tensorflow as tf
import time
from glob import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


st=time.time()
    
def askdas(i):
    image_path = "G:/Testing with inceptin v3/Custom-Image-Classification-using-Inception-v3-master/testing_image/"+i

    image_data = tf.io.gfile.GFile(image_path, 'rb').read()
    label_lines = [line.rstrip() for line
                   in tf.io.gfile.GFile("tf_files/retrained_labels.txt")]

    with tf.io.gfile.GFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.compat.v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
        print("--- %s seconds ---" % (time.time() - st))
        print('--------------------|||||||||||||||||||||------------------------------------')
          
def ninja():
    k=0
    pth ="G:/Testing with inceptin v3/Custom-Image-Classification-using-Inception-v3-master/testing_image/"
    for i in glob(pth+"*.jpg"):
           k=k+1
          
           print("image no.",k)
           z=i.split('\\')
           print('image name :-',z[1])
           askdas(z[1])
    for i in glob(pth+"*.jpeg"):
           k=k+1
           print('--------------------|||||||||||||||||||||------------------------------------')
           print("image no.",k)
           z=i.split('\\')
           print('image name :-',z[1])
           askdas(z[1])


ninja()  
