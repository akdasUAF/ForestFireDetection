import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

tf.disable_v2_behavior()

import glob2 as gb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys

def import_dataset(batch_size, class_names, target_size):
    img_dir = '/content/drive/MyDrive/IEEE_Flame/'
    #img_dir = 'C:\\Users\\nicol\\Downloads\\'
    train_ds = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
            img_dir + 'Training/', batch_size=batch_size, class_mode='categorical', classes = class_names,
        seed=123, shuffle=True, target_size=target_size, subset='training'
            )

    train_images, x_train_labels = next(train_ds)
    train_images = np.array(train_images)
    x_train_labels = np.array(x_train_labels)

    #validation_ds = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    #    img_dir + 'Training/', batch_size=batch_size, class_mode='categorical', 
    #    seed=123, shuffle=True, target_size=target_size, subset='validation'
    #    )
    validation_ds = train_ds

    validation_images, validation_labels = next(validation_ds)
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    test_ds = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        img_dir + 'Test/', batch_size=batch_size, class_mode='categorical', classes = class_names,
        seed=123, shuffle=True, target_size=target_size,
        )

    test_images, test_labels = next(test_ds)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    x_train, x_test, y_train , y_test = train_test_split(train_images, test_images, test_size=0.2)

    # x_train.shape: (25, 254, 254, 3), 193548 = 254*254*3
    trX, trY, teX, teY = x_train.reshape(x_train.shape[0], 193548).astype('float32')/255, y_train.reshape(x_train.shape[0], 193548).astype('float32')/255, x_test.reshape(x_test.shape[0], 193548).astype('float32')/255, y_test.reshape(x_test.shape[0], 193548).astype('float32')/255
    return (trX, trY, teX, teY)

class RBM(object):
    def __init__(self, input_size, output_size, learning_rate, batch_size):
        self.input_size = input_size #Size of the input layer
        self.output_size = output_size #Size of the hidden layer
        self.epochs = 1 #How many times we will update the weights 
        self.learning_rate = learning_rate #How big of a weight update we will perform 
        self.batch_size = batch_size #How many images will we "feature engineer" at at time 
        self.new_input_layer = None #Initalize new input layer variable for k-step contrastive divergence 
        self.new_hidden_layer = None
        self.new_test_hidden_layer = None
        
        #Here we initialize the weights and biases of our RBM
        #If you are wondering, the 0 is the mean of the distribution we are getting our random weights from. 
        #The .01 is the standard deviation.
        self.w = np.random.normal(0,.01,[input_size,output_size]) #weights
        self.hb = np.random.normal(0,.01,[output_size]) #hidden layer bias
        self.vb = np.random.normal(0,.01,[input_size]) #input layer bias (sometimes called visible layer)
        
        
        #Calculates the sigmoid probabilities of input * weights + bias
        #Here we multiply the input layer by the weights and add the bias
        #This is the phase that creates the hidden layer
    def prob_h_given_v(self, visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)
        
        #Calculates the sigmoid probabilities of input * weights + bias
        #Here we multiply the hidden layer by the weights and add the input layer bias
        #This is the reconstruction phase that recreates the original image from the hidden layer
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)
    
    #Returns new layer binary values
    #This function returns a 0 or 1 based on the sign of the probabilities passed to it
    #Our RBM will be utilizing binary features to represent the images
    #This function just converts the features we have learned into a binary representation 
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))
    
    def train(self, X):
        
        #Initalize placeholder values for graph
        _w = tf.placeholder(tf.float32, shape = [self.input_size, self.output_size])
        _vb = tf.placeholder(tf.float32, shape = [self.input_size])
        _hb = tf.placeholder(tf.float32, shape = [self.output_size])
        
        
        #initalize previous variables
        #we will be saving the weights of the previous and current iterations
        pre_w = np.random.normal(0,.01, size = [self.input_size,self.output_size])
        pre_vb = np.random.normal(0, .01, size = [self.input_size])
        pre_hb = np.random.normal(0, .01, size = [self.output_size])
        
        #initalize current variables
        #we will be saving the weights of the previous and current iterations
        cur_w = np.random.normal(0, .01, size = [self.input_size,self.output_size])
        cur_vb = np.random.normal(0, .01, size = [self.input_size])
        cur_hb = np.random.normal(0, .01, size = [self.output_size])
               
        #Plaecholder variable for input layer
        v0 = tf.placeholder(tf.float32, shape = [None, self.input_size])
         
        #pass probabilities of input * w + b into sample prob to get binary values of hidden layer
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb ))
        
        #pass probabilities of new hidden unit * w + b into sample prob to get new reconstruction
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        
        #Just get the probailities of the next hidden layer. We wont need the binary values. 
        #The probabilities here help calculate the gradients during back prop 
        h1 = self.prob_h_given_v(v1, _w, _hb)
        
        
        #Contrastive Divergence
        positive_grad = tf.matmul(tf.transpose(v0), h0) #input' * hidden0
        negative_grad = tf.matmul(tf.transpose(v1), h1) #reconstruction' * hidden1
        #(pos_grad - neg_grad) / total number of input samples 
        CD = (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0]) 
        
        #This is just the definition of contrastive divergence 
        update_w = _w + self.learning_rate * CD
        update_vb = _vb + tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + tf.reduce_mean(h0 - h1, 0)
        
        #MSE - This is our error function
        err = tf.reduce_mean(tf.square(v0 - v1))
        
        #Will hold new visible layer.
        errors = []
        hidden_units = []
        reconstruction = []
        
        test_hidden_units = []
        test_reconstruction=[]
        
        
        #The next four lines of code intitalize our Tensorflow graph and create mini batches
        #The mini batch code is from cognitive class. I love the way they did this. Just giving credit! 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                for start, end in zip(range(0, len(X), self.batch_size), range(self.batch_size, len(X), self.batch_size)):
                    batch = X[start:end] #Mini batch of images taken from training data
                    
                    #Feed in batch, previous weights/bias, update weights and store them in current weights
                    cur_w = sess.run(update_w, feed_dict = {v0:batch, _w:pre_w , _vb:pre_vb, _hb:pre_hb})
                    cur_hb = sess.run(update_hb, feed_dict = {v0:batch, _w:pre_w , _vb:pre_vb, _hb:pre_hb})
                    cur_vb = sess.run(update_vb, feed_dict = {v0:batch, _w:pre_w , _vb:pre_vb, _hb:pre_hb})
                    
                    #Save weights 
                    pre_w = cur_w
                    pre_hb = cur_hb
                    pre_vb = cur_vb
                
                #At the end of each iteration, the reconstructed images are stored and the error is outputted 
                reconstruction.append(sess.run(v1, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb}))        
                print('Learning Rate: {}:  Batch Size: {}:  Hidden Layers: {}: Epoch: {}: Error: {}:'.format(self.learning_rate, self.batch_size, 
                                                                                                             self.output_size, (epoch+1),
                                                                                                            sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})))
            
            #Store final reconstruction in RBM object
            self.new_input_layer = reconstruction[-1]
            
            #Store weights in RBM object
            self.w = pre_w
            self.hb = pre_hb
            self.vb = pre_vb
    
    #This is used for Contrastive Divergence.
    #This function makes the reconstruction your new input layer. 
    def rbm_output(self, X):
        #sess = tf.Session()
        
        #input_x = tf.constant(X).eval(session=sess)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        _vb = tf.constant(self.vb)

        _w = tf.cast(_w, tf.float32)
        _hb = tf.cast(_hb, tf.float32)
        _vb = tf.cast(_vb, tf.float32)
        out = tf.nn.sigmoid(tf.matmul(X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)
    
def train_DBN(RBM_hidden_size, learning_rate, batch_size, class_names, target_size):
  (trX, _, teX, _) = import_dataset(batch_size, class_names, target_size)
  input_size = trX.shape[1] #input layer size of original image data

  rbm_list = [] #This will hold all of the RBMs used in our DBN

  #Creates 3 RBMs
  for layer in RBM_hidden_size:
      rbm_list.append(RBM(input_size, layer, learning_rate, 32))
      input_size = layer

  #Initalize input layer variables 
  inpX = trX                
  test_inpx = teX

  #This loop is the DBN. Each rbm is trained here.
  #At the end of training, the hidden layer of the RBM is used as input
  #For the next layer of the DBN. 
  for i, rbm in enumerate(rbm_list):
    rbm_outputs = []
    rbm_test_outputs = []
    print('Input Shape: ', inpX.shape)
    print('Layer: ',(i+1))

    rbm.train(inpX, teX)
    #inpX = tf.cast(inpX, tf.float32)
    inpX = rbm.rbm_output(inpX)
    test_inpx = rbm.rbm_output(test_inpx)
    rbm_outputs.append(inpX)
    rbm_test_outputs.append(test_inpx)

    print('Output Shape: ', inpX.shape)
  return rbm_list

def fwd_DBN(rbm_list, features):
  # input_size = features.shape[1] #input layer size of original image data

  #Initalize input layer variables 
  inpX = features                

  #This loop is the DBN. Each rbm is trained here.
  #At the end of training, the hidden layer of the RBM is used as input
  #For the next layer of the DBN. 
  for i, rbm in enumerate(rbm_list):
    rbm_outputs = []
    print('Input Shape: ', inpX.shape)
    print('Layer: ',(i+1))

    inpX = rbm.rbm_output(inpX)
    rbm_outputs.append(inpX)

    print('Output Shape: ', inpX.shape)
  return inpX



def main():
    # Train Setup
    target_size = (254, 254)
    batch_size = 32 # Default is 32
    class_names = ['Fire', 'No_Fire']
    (trX, trY, teX, teY) = import_dataset(batch_size, class_names, target_size)

    RBM_hidden_size = [600, 500, 100] #Three hidden layer sizes for our three layer DBN
    learning_rate = .01 
    
    rmb_list = train_DBN(RBM_hidden_size, learning_rate, batch_size, class_names, target_size)

    train_features = fwd_DBN(rmb_list, trX)
    train_labels = trY
    clf = LogisticRegression()
    #train_labels = train_labels.reshape(train_labels.shape[0], )
    #train_labels = train_labels.reshape((100,))

    clf.fit(train_features, train_labels.argmax(axis=1))
    #trX, trY, teX, teY = 0, 0, 0, 0
    import joblib
    filename = './Models/weights/logistic_regression_model.pkl'
    joblib.dump(clf, filename)

    return 0


if __name__ == '__main__':
    sys.exit(main())
