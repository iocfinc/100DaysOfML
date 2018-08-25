# First Importing the bottle neck features
bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']
# I just want to know the shape of the output of the Inceptionv# data
print(train_InceptionV3.shape[1:])
# Then creating the model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Note: I added a re-import because I was actually doing this project by parts and I don't want to run back to the earlier codes.

TF_InceptionV3 = Sequential()
TF_InceptionV3.add(GlobalAveragePooling2D(input_shape = train_InceptionV3.shape[1:]))
TF_InceptionV3.add(Dense(500, activation='relu'))
TF_InceptionV3.add(dropout(0.2)) # To avoid overfitting
TF_InceptionV3.add(Dense(133, activation='softmax')) # Softmax to provide the final output.


# Model summary
TF_InceptionV3.summary()
# Model Compilation
TF_InceptionV3.compile(optimizer=optimizers.RMSprop(lr = 1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# Model training
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.TF_InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)

History = TF_InceptionV3.fit(train_InceptionV3, train_targets, 
          validation_data=(valid_InceptionV3, valid_targets),
          epochs=10, batch_size=20, callbacks=[checkpointer], verbose=1)
# Load the model
TF_InceptionV3.load_weights('saved_models/weights.best.TF_InceptionV3.hdf5')
# Test model
# get index of predicted dog breed for each image in test set
TF_InceptionV3_predictions = [np.argmax(TF_InceptionV3.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

# report test accuracy
test_accuracy = 100*np.sum(np.array(TF_InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(TF_InceptionV3_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

# Prediction model
def TF_InceptionV3_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_TF_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = TF_InceptionV3.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

# Algorithm for Human Dog classifier and Dog breed matcher
import cv2
from keras.applications.resnet50 import ResNet50
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def TF_Human_Dog_Breed_Checker(img_path):
    if face_detector(img_path):
        print("Dog Detected")

    ('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Female_German_Shepherd.jpg/330px-Female_German_Shepherd.jpg')

# face_detector(img_path): Will return True if face is detected
# dog_detector(img_path) : Will return True if dog is detected
# dog breed classifier: Will ouput Breed

TF_InceptionV3.add(BatchNormalization())
TF_InceptionV3.add(activation='Relu')