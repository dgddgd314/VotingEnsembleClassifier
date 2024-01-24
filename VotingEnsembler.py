import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class CNNVoteEnsembleClassifier():

# When defining this model, we will get two parameters : list of model and ImageDataGenerator. (because it is CNN)
  def __init__(self, model_list, datagen = 'None'):
    
    # model_list include every set of model.
    self.model_list = model_list
    
    # because datagen generates random image, we have to get datagenerator from the start.
    if datagen == 'None':
        self.datagen = ImageDataGenerator(
            rescale = 1/255
        )
    else:
      self.datagen = datagen
      
    # if this model is not fitted, raise Error.
    self.__fit__ = False

  # from this fit function, you can fit the models.
  def fit(self, test_data, test_labels, fit_base_estimators = False, *args, **kwargs):
    
    # because this is fitted.
    self.__fit__ = True
    
    # if fit_base_estimators is True, fit. if not; just remain it.
    if fit_base_estimators == True:
      pass # soon update : this part will be used at learning each models. for **kwargs, those will support other parameters for AI : like batches...
    
    
    # if it is false, remain it as it is.
    else :
      pass
  
  ## about accuracy, this will be disappeared. this is over-functioned.
  # calculating cross entropy
  def categorical_crossentropy(self, y_true, y_pred):
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  
    N = y_pred.shape[0]  

    cross_entropy = -np.sum(y_true * np.log(y_pred + epsilon)) / N

    return cross_entropy

  def accuracy(self, y_true, y_pred):
    # 각 예측에서 가장 높은 확률을 갖는 클래스 선택
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # 정확하게 예측된 비율 계산
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples

    return accuracy

  # loss와 정확도 표시
  def show_accuracy(self, imglabel):
    print(f'loss : {self.categorical_crossentropy(imglabel, self.prediction)}, accuracy : {self.accuracy(imglabel, self.prediction)}')
    
  ## until here.

  # test_data(4차원)이 들어오면 이를 예측.
  def predict(self, test_data):
    if self._fit__ == False:
      raise Exception
    result_list = []

    for model in self.model_list:
      index = np.zeros(len(test_data))
      test_generator = self.datagen.flow(test_data, index, batch_size = 32, shuffle = False)
      result = model.predict(test_generator)
      result_list.append(result)

    result_array = np.array(result_list)

    prediction = []

    for i in range(result_array.shape[1]):
      predictions_for_sample = result_array[:, i, :]
      prediction.append(np.mean(predictions_for_sample, axis = 0))

    self.prediction = prediction
    print(prediction)
    
  # this function will be also disappeared soon.
  def show_accuracy_score(self, test_data, test_labels):
    result_list = []

    for model in self.model_list:
      test_generator = self.datagen.flow(test_data, test_labels, batch_size = 32, shuffle = False)
      result = model.predict(test_generator)
      result_list.append(result)

    result_array = np.array(result_list)

    prediction = []

    for i in range(result_array.shape[1]):
      predictions_for_sample = result_array[:, i, :]
      prediction.append(np.mean(predictions_for_sample, axis = 0))

    self.prediction = prediction
    
    self.show_accuracy(test_labels)