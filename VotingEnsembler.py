import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class VoteEnsembleClassifier():

  def __init__(self, model_list):
    self.model_list = model_list

  # 여기서는 test_data, test_labes에 대해 prediction matrix를 추출한다는 의미
  def fit(self, test_data, test_labels, datagen = 'None'):
    if datagen == 'None':
        datagen = ImageDataGenerator(
            rescale = 1/255
        )
    result_list = []

    for model in self.model_list:
      test_generator = datagen.flow(test_data, test_labels, batch_size = 32, shuffle = False)
      result = model.predict(test_generator)
      result_list.append(result)

    result_array = np.array(result_list)

    prediction = []

    for i in range(result_array.shape[1]):
      predictions_for_sample = result_array[:, i, :]
      prediction.append(np.mean(predictions_for_sample, axis = 0))

    self.prediction = prediction

  def categorical_crossentropy(self, y_true, y_pred):
    epsilon = 1e-15  # 아주 작은 값, 로그의 분모가 0이 되는 것을 방지하기 위해 추가됨
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 예측값을 0에서 1 사이로 클리핑
    N = y_pred.shape[0]  # 배치 크기

    # 교차 엔트로피 계산
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

  # test_data(4차원)이 들어오면 이를 예측.
  def predict(self, test_data):
    result_list = []

    for model in self.model_list:
      index = np.zeros(len(test_data))
      test_generator = datagen.flow(test_data, index, batch_size = 32, shuffle = False)
      result = model.predict(test_generator)
      result_list.append(result)

    result_array = np.array(result_list)

    prediction = []

    for i in range(result_array.shape[1]):
      predictions_for_sample = result_array[:, i, :]
      prediction.append(np.mean(predictions_for_sample, axis = 0))

    self.prediction = prediction
    print(prediction)