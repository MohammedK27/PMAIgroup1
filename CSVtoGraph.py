import pandas as pd
import matplotlib.pyplot as plt

#load data
data = pd.read_csv("OriginalCNN.csv")
altcnn = pd.read_csv("3LayerCNN.csv")
og20epoch = pd.read_csv("OriginalCNN20epochs.csv") # new base graph
altcnn20epoch = pd.read_csv("3LayerCNN20epoch.csv")
#optimiser comparison
SGDlr0001 = pd.read_csv("SGDlr0001.csv") # SDG with learning rate of 0.001
SGDwithM = pd.read_csv("SGDwithM.csv") #SGD with momentum
adamopt = pd.read_csv("adamopt.csv")  #adam optimiser
#data augmentation
withaug = pd.read_csv("withAugmentation.csv") # original with data augmentation during training
#dropout
dropout5 = pd.read_csv("dropout5.csv")
dropout10 = pd.read_csv("dropout10.csv")
dropout15 = pd.read_csv("dropout15.csv")
dropout20 = pd.read_csv("dropout20.csv")

dropout30 = pd.read_csv("dropout30.csv")
dropout35 = pd.read_csv("dropout35.csv")
dropout40 = pd.read_csv("dropout40.csv")
dropout45 = pd.read_csv("dropout45.csv")
dropout50 = pd.read_csv("dropout50.csv")

#plot graphs
plt.figure()
plt.plot(dropout5['Epoch'], dropout5['Test Accuracy'], label='Dropout of 5%')
plt.plot(dropout10['Epoch'], dropout10['Test Accuracy'], label='Dropout of 10%')
plt.plot(dropout15['Epoch'], dropout15['Test Accuracy'], label='Dropout of 15%')
plt.plot(dropout20['Epoch'], dropout20['Test Accuracy'], label='Dropout of 20%')
plt.plot(og20epoch['Epoch'], og20epoch['Test Accuracy'], label='Dropout of 25%')
plt.plot(dropout30['Epoch'], dropout30['Test Accuracy'], label='Dropout of 30%')
plt.plot(dropout35['Epoch'], dropout35['Test Accuracy'], label='Dropout of 35%')
plt.plot(dropout40['Epoch'], dropout40['Test Accuracy'], label='Dropout of 40%')
plt.plot(dropout45['Epoch'], dropout45['Test Accuracy'], label='Dropout of 45%')
plt.plot(dropout50['Epoch'], dropout50['Test Accuracy'], label='Dropout of 50%')
plt.xticks(dropout5['Epoch'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy when changing dropout percentage')
plt.legend()
plt.show()

plt.figure()
plt.plot(og20epoch['Epoch'], og20epoch['Test Accuracy'], label='Original Accuracy')
plt.plot(og20epoch['Epoch'], og20epoch['Train Accuracy'], label='Original Train Accuracy')
plt.plot(withaug['Epoch'], withaug['Test Accuracy'], label='Accuracy With Data Augmentation')
plt.plot(withaug['Epoch'], withaug['Train Accuracy'], label='Train Accuracy With Data Augmentation')
plt.xticks(og20epoch['Epoch'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy when Introducing Data Augmentation')
plt.legend()
plt.show()

plt.figure()
plt.plot(data['Epoch'], data['Train Loss'], label="Training Loss")
plt.plot(data['Epoch'], data['Test Loss'], label="Testing Loss")
plt.xticks(data['Epoch'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
#plt.show()

plt.figure()
plt.plot(data['Epoch'], data['Train Accuracy'], label='Training Accuracy')
plt.plot(data['Epoch'], data['Test Accuracy'], label='Test Accuracy')
plt.xticks(data['Epoch'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Epochs')
plt.legend()
#plt.show()

plt.figure()
plt.plot(data['Epoch'], data['Test Accuracy'], label='Original Accuracy')
plt.plot(altcnn['Epoch'], altcnn['Test Accuracy'], label='Accuracy with a Third Layer')
plt.xticks(data['Epoch'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Effect of a Third Convolutional Layer on Accuracy')
plt.legend()
#plt.show()

plt.figure()
plt.plot(og20epoch['Epoch'], og20epoch['Test Accuracy'], label='Original Accuracy')
plt.plot(altcnn20epoch['Epoch'],altcnn20epoch['Test Accuracy'], label='Accuracy with a Third Layer')
plt.xticks(og20epoch['Epoch'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Effect of a Third Convolutional Layer on Accuracy')
plt.legend()

#optimiser comparison
plt.figure()
plt.plot(og20epoch['Epoch'], og20epoch['Test Accuracy'], label='SGD, learning rate 0.01')
plt.plot(SGDlr0001['Epoch'], SGDlr0001['Test Accuracy'], label='SGD, learning rate 0.001')
plt.plot(SGDwithM['Epoch'], SGDwithM['Test Accuracy'], label='SGD with Momentum, learning rate 0.001')
plt.plot(adamopt['Epoch'], adamopt['Test Accuracy'], label='Adam optimiser, learning rate 0.001')
plt.plot(adamopt['Epoch'], adamopt['Train Accuracy'], label='Adam Optimiser, Training accuracy')
plt.xticks(og20epoch['Epoch'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Effects of different optimisers')
plt.legend()
#plt.show()