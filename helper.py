import numpy as np
import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
from glob import glob
import time 


def gen_test_output(sess, logits, is_training, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    #shape_list = []
    
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.jpg')):
        shape_list_ = scipy.misc.imread(image_file).shape
     #   shape_list.append(shape_list_) 
        
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        start = time.time()
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {is_training: False, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.3).reshape(image_shape[0], image_shape[1], 1)#0.5
        #print (segmentation)
        #print (type(segmentation))
        #print (max(segmentation))
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        end = time.time()
        print("time cost:",end-start)

        yield os.path.basename(image_file), np.array(street_im), shape_list_


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, is_training, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, is_training, input_image, os.path.join(data_dir, 'data_road/training'), image_shape)
    for name, image , shape in image_outputs:
        image = scipy.misc.imresize(image,shape)
        scipy.misc.imsave(os.path.join(output_dir, name), image)
