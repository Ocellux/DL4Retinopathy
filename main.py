import pickle
from flask import Flask, request, jsonify
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
#from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
##creating a flask app and naming it "app"
app = Flask('app')

@app.route('/test', methods=['GET'])
def test():
    return 'Pinging Model Application!!'

@app.route('/home')
def upload_file():
   return render_template('ecarebetics.html')
	
@app.route('/uploader', methods = ['POST'])
def getresults():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      #print (f)
      HEIGHT = 512#1024#3216/2
      WIDTH = 512#1024#2136/2
      batch_size=1
      predicted_class_list=[]
      new_model = tf.keras.models.load_model("ecarebetics_model.h5")
      #my_image = load_img(secure_filename(f.filename), target_size=(WIDTH, HEIGHT))
      my_image = f.reshape((1, f.shape[0], f.shape[1], f.shape[2]))
      data=np.array(my_image)/255 # convert to an np array and rescale images
      y_pred = new_model.predict(data,batch_size=batch_size, verbose=0 )
      trials=len (y_pred)
      for i in range(0,trials):
          predicted_class=y_pred[i].argmax() # get index of highest probability
          predicted_class_list.append(predicted_class)
          #print (predicted_class) # print file name and class prediction

      #f.save(secure_filename(f.filename))
      print (predicted_class_list)
      result = {
        'mpg_prediction': list(predicted_class_list)
        }
    
      #return 'file uploaded successfully'
      return jsonify(result)



@app.route('/predict', methods=['POST'])
def predict():
    image = request.get_json()
    print(image)
    batch_size=1
    predicted_class_list=[]
    new_model = tf.keras.models.load_model("ecarebetics_model.h5")
    y_pred = new_model.predict(image,batch_size=batch_size, verbose=0 )
    trials=len (y_pred)
    for i in range(0,trials):
        predicted_class=y_pred[i].argmax() # get index of highest probability
        predicted_class_list.append(predicted_class)
        print (predicted_class) # print file name and class prediction

    result = {
        'mpg_prediction': list(predicted_class_list)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
