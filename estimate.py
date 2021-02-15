import matplotlib.pyplot as plt
from tensorflow import keras

PATH = './Amerilike_6ch'

model = keras.models.load_model(PATH)

# Predictions
print('INFO: Predict labels')
Train_predicted_labels = pd.DataFrame().reindex_like(Train_Labels)
Train_predicted_labels.iloc[:, :] = estimator.predict(np.array(Train[0]), batch_size=600)

for i in range(1, 7):
    plt.figure(figsize=(16, 4))
    plt.plot(Train_Labels['Right_Hand_channel{}'.format(i)])
    plt.plot(Train_predicted_labels['Right_Hand_channel{}'.format(i)])

    plt.plot(Labels['Right_Hand_channel{}'.format(i)].iloc[:int(len(Labels) * 3 / 5)] / np.max(
        Labels['Right_Hand_channel{}'.format(i)]) * 10)
    plt.show()