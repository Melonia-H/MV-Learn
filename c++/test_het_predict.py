import numpy as np
import tensorflow as tf
from lanenet_model import lanenet_cluster
import cv2

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  #dims_expander = tf.expand_dims(float_caster, 0)
  #dims_expamder = np.array(float_caster).reshape(1,input_height, input_width, 3)
  #resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  #print(resized)
  
  normalized = tf.divide(tf.subtract(float_caster, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

if __name__ == "__main__":
  file_name = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/data/0001.png"
  model_file = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/tusimple_model/tusimple_lanenet/other/lanenet.pb"
  #model_file ='/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/tusimple_model/tusimple_lanenet/tusimple_lanenet_enet_2019-06-26-14-54-26.ckpt-21000'
  
  input_height = 256
  input_width = 512
  input_mean = 0
  input_std = 255
  input_layer1 = "input_tensor"
  input_layer2 = "net_phase"
  output_layer1 = "lanenet_model/pix_embedding_relu"
  output_layer2 = "lanenet_model/inference/LaneNetSeg/fullconv/conv2d_transpose"

  phase = False

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
  ts = []
  for i in range(0,8):
    ts.append(t)
  
  input_name = "import/" + input_layer1
  input_name1= "import/" + input_layer2
  output_name1= "import/" + output_layer1
  output_name2 = "import/" + output_layer2
  
  input_operation = graph.get_operation_by_name(input_name)
  input_operation1 = graph.get_operation_by_name(input_name1)
  output_operation1 = graph.get_operation_by_name(output_name1)
  output_operation2 = graph.get_operation_by_name(output_name2)

  with tf.Session(graph=graph) as sess:
    results1 = sess.run(output_operation1.outputs[0], {
        input_operation.outputs[0]: ts,
        input_operation1.outputs[0]: phase,
    })

    results2 = sess.run(output_operation2.outputs[0], {
        input_operation.outputs[0]: ts,
        input_operation1.outputs[0]: phase,
    })
    
  instance_seg_image = np.squeeze(results1)   
  results2 = np.squeeze(results2)

  binary_seg_image = tf.nn.softmax(logits=results2)
  binary_seg_image = tf.argmax(binary_seg_image, axis=-1)

  print('instance_seg_image')
  print(instance_seg_image.shape)

  print('binary_seg_image')
  print(binary_seg_image)

  cluster = lanenet_cluster.LaneNetCluster()

  mask = np.random.randn(binary_seg_image[0].shape[0], binary_seg_image[0].shape[1]) > 0.5
  bi = binary_seg_image[0] * mask

  image = cv2.imread(file_name, cv2.IMREAD_COLOR)
  gt_image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
   
  mask_image, lane_coordinate, cluster_index, labels = cluster.get_lane_mask(binary_seg_ret=bi,
                                           instance_seg_ret=instance_seg_image[0], gt_image=gt_image)

  print('binary_seg_image')
  print(mask_image)
  
  print('lane_coordinate')
  print(lane_coordinate)
  
  print('cluster_index')
  print(cluster_index)
  
  print('labels')
  print(labels)

  


  



















  
