# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T16:50:17.192477Z","iopub.execute_input":"2023-04-23T16:50:17.193842Z","iopub.status.idle":"2023-04-23T16:50:33.100319Z","shell.execute_reply.started":"2023-04-23T16:50:17.193794Z","shell.execute_reply":"2023-04-23T16:50:33.099087Z"}}
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, InputLayer, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T16:51:12.008713Z","iopub.execute_input":"2023-04-23T16:51:12.009601Z","iopub.status.idle":"2023-04-23T16:51:19.091512Z","shell.execute_reply.started":"2023-04-23T16:51:12.009565Z","shell.execute_reply":"2023-04-23T16:51:19.090400Z"}}
img_size = 128
batch_size = 128
def preprocess_image(image, img_size=(img_size, img_size)):
    # Resize image
    image = tf.image.resize(image, img_size)
    
    # Apply median filter
    filter_shape = (2, 2)
    padding = "SYMMETRIC"
    image = tfa.image.median_filter2d(image, padding="CONSTANT", constant_values=0)
    
    # Normalize pixel values to be between 0 and 1
    image = image / 255.0
    
    return image

train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: median_filter(x, size=2, mode='constant', cval=0.0),
    rescale = 1/255., brightness_range = [0.5,1.5], zoom_range = 0.2, 
    width_shift_range = 0.15, height_shift_range = 0.15, horizontal_flip = True 
)
val_datagen = ImageDataGenerator(rescale = 1/255.)
test_datagen = ImageDataGenerator(rescale = 1/255.)


train_generator = train_datagen.flow_from_directory('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN',
                                                   target_size = (img_size, img_size),
                                                   batch_size = batch_size,
                                                   shuffle=True,
                                                   class_mode='categorical')

val_generator = val_datagen.flow_from_directory('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TEST_SIMPLE',
                                                   target_size = (img_size, img_size),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode='categorical')

test_generator = test_datagen.flow_from_directory('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TEST',
                                                   target_size = (img_size, img_size),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode = 'categorical')

# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T16:51:19.095068Z","iopub.execute_input":"2023-04-23T16:51:19.095774Z","iopub.status.idle":"2023-04-23T16:51:24.490126Z","shell.execute_reply.started":"2023-04-23T16:51:19.095735Z","shell.execute_reply":"2023-04-23T16:51:24.488933Z"}}
input_layer = Input(shape=(128, 128, 3))

x = Conv2D(6, kernel_size=(1,1), strides=(1,1))(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(16, kernel_size=(5,5), strides=(1,1))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, kernel_size=(5,5), strides=(1,1))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, kernel_size=(5,5), strides=(1,1))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, kernel_size=(4,4), strides=(1,1))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Dropout(0.2)(x)

x = Flatten()(x) 
output_layer = Dense(4, activation='softmax')(x)

model = Model(inputs=input_layer,outputs=output_layer)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T16:51:24.491735Z","iopub.execute_input":"2023-04-23T16:51:24.492162Z","iopub.status.idle":"2023-04-23T16:51:24.551674Z","shell.execute_reply.started":"2023-04-23T16:51:24.492119Z","shell.execute_reply":"2023-04-23T16:51:24.550715Z"}}
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T16:51:58.141671Z","iopub.execute_input":"2023-04-23T16:51:58.142368Z","iopub.status.idle":"2023-04-23T16:51:58.150231Z","shell.execute_reply.started":"2023-04-23T16:51:58.142312Z","shell.execute_reply":"2023-04-23T16:51:58.149163Z"}}
train_generator.class_indices

# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T16:51:59.051236Z","iopub.execute_input":"2023-04-23T16:51:59.051643Z","iopub.status.idle":"2023-04-23T16:51:59.275137Z","shell.execute_reply.started":"2023-04-23T16:51:59.051607Z","shell.execute_reply":"2023-04-23T16:51:59.274063Z"}}
#decayed_lr = tf.train.exponential_decay(learning_rate,
                                        #global_step, 10000,
                                        #0.95, staircase=True)
#opt = tf.train.AdamOptimizer(decayed_lr, epsilon=adam_epsilon)
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9
grad_decay = 0.1
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    ),
    beta_1=1.0 - grad_decay  # Set beta_1 to 0.9 for 0.1 gradient decay
)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T11:00:59.265857Z","iopub.execute_input":"2023-04-23T11:00:59.266167Z","iopub.status.idle":"2023-04-23T11:00:59.273361Z","shell.execute_reply.started":"2023-04-23T11:00:59.266134Z","shell.execute_reply":"2023-04-23T11:00:59.272277Z"}}
"""history = model.fit_generator(
    train_generator,
    validation_data = val_generator,
    epochs = 30
)"""


# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T16:52:03.421133Z","iopub.execute_input":"2023-04-23T16:52:03.422254Z"}}
from tensorflow.keras.models import load_model

# model = load_model("/kaggle/input/wbc-cnn-model/model")


history = model.fit_generator(
    train_generator,
    validation_data = val_generator,
    epochs = 20
)
model.save("./model")

# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T18:08:41.665451Z","iopub.execute_input":"2023-04-23T18:08:41.665863Z","iopub.status.idle":"2023-04-23T18:08:41.690870Z","shell.execute_reply.started":"2023-04-23T18:08:41.665829Z","shell.execute_reply":"2023-04-23T18:08:41.689116Z"}}
history_dict = history.history
train_acc = history_dict['loss']
val_acc = history_dict['val_loss']
epochs = range(1, len(history_dict['loss'])+1)
plt.plot(epochs, train_acc,'b', label='Training error')
plt.plot(epochs, val_acc,'b', color="orange", label='Validation error')
plt.title('Training and Validation error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-23T18:08:44.454268Z","iopub.execute_input":"2023-04-23T18:08:44.455473Z","iopub.status.idle":"2023-04-23T18:08:44.481862Z","shell.execute_reply.started":"2023-04-23T18:08:44.455426Z","shell.execute_reply":"2023-04-23T18:08:44.480215Z"}}
history_dict = history.history
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy'])+1)
plt.plot(epochs, train_acc,'b', label='Training accuracy')
plt.plot(epochs, val_acc,'b', color="orange", label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# %% [code]
y_preds = model.predict_generator(test_generator)

# %% [code]
test_generator = test_datagen.flow_from_directory('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TEST',
                                                   target_size = (img_size, img_size),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode = 'categorical')

x, y = test_generator.next()
y_true = y
for i in range(2487//128):
    x, y = test_generator.next()
    y_true = np.concatenate([y_true, y], axis = 0)
    
print(y_true)

# %% [code]
y_true = np.argmax(y_true, axis = 1)
y_preds = np.argmax(y_preds, axis = 1)

print(y_true, y_preds)

# %% [code]
import seaborn as sns
import matplotlib.pyplot as plt     
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_true, y_preds)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL'])
ax.yaxis.set_ticklabels(['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL'])

# %% [code]
from sklearn.metrics import classification_report

print(classification_report(y_true, y_preds, target_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']))

# %% [code]


# %% [code]
