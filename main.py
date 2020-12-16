# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main-window.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gan_model import MMDStatistic
from operations import Operation

# Generator Imports
import tensorflow as tf
import sklearn.preprocessing
import scipy.ndimage

# Encoder imports
from mymodel import LSTMEncoder
from mymodel import AE_opt
import torch
import torch.optim as optim
import torch.nn.functional as F


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        # Top bar
        self.actionLoad_npy = QtWidgets.QAction(MainWindow)
        self.actionLoad_npy.setObjectName("actionLoad_npy")
        self.actionLoad_npy.triggered.connect(self.loadNPY)


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Operations save for generating multiple signals with similar operations after the generation phase
        self.operations = []
        self.amtToGenerate = 2 # Amount of generation
        self.generateManyButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateManyButton.setGeometry(QtCore.QRect(10, 530, 121, 30))
        self.generateManyButton.setObjectName("generateManyButton")
        self.generateManyButton.setText("Generate Batch")
        self.generateManyButton.clicked.connect(self.pressGenerateManyButton)

        self.generateManyLabel = QtWidgets.QLabel(self.centralwidget)
        self.generateManyLabel.setGeometry(QtCore.QRect(200, 530, 300, 22))
        self.generateManyLabel.setText("# of signals to generate: " + str(self.amtToGenerate))
        self.generateManyLabel.setObjectName("generateManyLabel")

        self.generateNumField = QtWidgets.QLineEdit(self.centralwidget)
        self.generateNumField.setGeometry(QtCore.QRect(140, 530, 60, 30))
        self.generateNumField.setText(str(self.amtToGenerate))
        self.generateNumField.editingFinished.connect(self.numFieldEdited)


        # Generator run-through
        self.generateButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateButton.setGeometry(QtCore.QRect(530, 470, 121, 51))
        self.generateButton.setObjectName("generateButton")
        self.generateButton.clicked.connect(self.pressGenerateButton)
        # Generator setup
        self.generator_model = tf.keras.models.load_model('./gan_weights/generator_200e_20drop.h5')
        self.gen_norm_value = 2173



        # Plot area elements
        self.plotLabel = QtWidgets.QLabel(self.centralwidget)
        self.plotLabel.setGeometry(QtCore.QRect(10, 110, 511, 411))
        self.plotLabel.setText("")
        self.plotLabel.setScaledContents(True) # Fit to label
        self.plotLabel.setObjectName("plotLabel")
        self.plotTitle = ""

        # Plot variables
        # df = pd.read_csv("testingData/normal_valid.csv") # TEMP
        # arr = df.iloc[0][:-1] # TEMP
        self.x_values = np.linspace(0, 187, 187)
        self.y_values = np.linspace(0, 187, 187)


        # Encoder run-through
        self.verifyButton = QtWidgets.QPushButton(self.centralwidget)
        self.verifyButton.setGeometry(QtCore.QRect(660, 470, 121, 51))
        self.verifyButton.setObjectName("verifyButton")
        self.verifyButton.clicked.connect(self.pressVerifyButton)
        # Encoder setup
        self.autoEncoder_model = LSTMEncoder()
        self.autoEncoder_optim = optim.Adam(self.autoEncoder_model.parameters(), lr=AE_opt.lr)
        self.autoEncoder_checkpoint = torch.load("./kllstm_epoch_999.pt", map_location=torch.device('cpu'))
        self.autoEncoder_model.load_state_dict(self.autoEncoder_checkpoint)
        self.autoEncoder_model.eval()

        # slider1 elements: Slider allows you to select individual datapoints
        self.sliderLabel1 = QtWidgets.QLabel(self.centralwidget)
        self.sliderLabel1.setGeometry(QtCore.QRect(530, 420, 160, 22))
        self.sliderLabel1.setText("Element Selected: 0")
        self.sliderLabel1.setObjectName("sliderLabel")

        self.slider1 = QtWidgets.QSlider(self.centralwidget)
        self.slider1.setGeometry(QtCore.QRect(530, 440, 250, 22))
        self.slider1.setOrientation(QtCore.Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(186) # Select any of the 187 datapoints
        self.slider1.setValue(0)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider1.setTickInterval(5)
        self.slider1.setObjectName("slider1")
        self.slider1.valueChanged.connect(self.slider1ValueChange) # Update label when value changed
        self.slider1.sliderReleased.connect(self.slider1Released) # Update label only when slider released

        # sliderSpread elements: Slider allows you to select a spread of datapoints
        self.sliderSpreadLabel = QtWidgets.QLabel(self.centralwidget)
        self.sliderSpreadLabel.setGeometry(QtCore.QRect(530, 370, 160, 22))
        self.sliderSpreadLabel.setText("Spread: 0")
        self.sliderSpreadLabel.setObjectName("sliderSpreadLabel")

        self.sliderSpread = QtWidgets.QSlider(self.centralwidget)
        self.sliderSpread.setGeometry(QtCore.QRect(530, 390, 250, 22))
        self.sliderSpread.setOrientation(QtCore.Qt.Horizontal)
        self.sliderSpread.setMinimum(0)
        self.sliderSpread.setMaximum(10)  # Select up to a spread of 5 on each side
        self.sliderSpread.setValue(0)
        self.sliderSpread.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderSpread.setTickInterval(1)
        self.sliderSpread.setObjectName("sliderSpread")
        self.sliderSpread.valueChanged.connect(self.sliderSpreadValueChange)  # Update label when value changed
        # self.sliderSpread.sliderReleased.connect(self.sliderSpread1Released)  # Update label only when slider released

        self.currentlySelectedPoint = 0 # The currently selected point will be highlighted in red on the scatter
        self.spreadPoints = [0]

        # Buttons that allow you to adjust the height of the selected point
        self.adjustAmt = 0.0

        self.upAdjustButton = QtWidgets.QPushButton(self.centralwidget)
        self.upAdjustButton.setGeometry(QtCore.QRect(530, 260, 40, 30))
        self.upAdjustButton.setObjectName("upAdjustButton")
        self.upAdjustButton.clicked.connect(self.pressUpAdjustButton)
        self.upAdjustButton.setIcon(QtGui.QIcon('icons/up_arrow.png'))

        self.downAdjustButton = QtWidgets.QPushButton(self.centralwidget)
        self.downAdjustButton.setGeometry(QtCore.QRect(530, 290, 40, 30))
        self.downAdjustButton.setObjectName("downAdjustButton")
        self.downAdjustButton.clicked.connect(self.pressDownAdjustButton)
        self.downAdjustButton.setIcon(QtGui.QIcon('icons/down_arrow.png'))

        self.sliderAdjustLabel = QtWidgets.QLabel(self.centralwidget)
        self.sliderAdjustLabel.setGeometry(QtCore.QRect(530, 320, 160, 22))
        self.sliderAdjustLabel.setText("Inc/Dec Amount: 0.0")
        self.sliderAdjustLabel.setObjectName("sliderAdjustLabel")

        self.sliderAdjust = QtWidgets.QSlider(self.centralwidget)
        self.sliderAdjust.setGeometry(QtCore.QRect(530, 340, 250, 22))
        self.sliderAdjust.setOrientation(QtCore.Qt.Horizontal)
        self.sliderAdjust.setMinimum(0)
        self.sliderAdjust.setMaximum(10)  # Select up to increment of 10 (convert to 1.0)
        self.sliderAdjust.setValue(0)
        self.sliderAdjust.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderAdjust.setTickInterval(1) # 1 = 0.1 increment
        self.sliderAdjust.setObjectName("sliderAdjust")
        self.sliderAdjust.valueChanged.connect(self.sliderAdjustValueChange)

        # Smoothing elements
        self.smoothButton = QtWidgets.QPushButton(self.centralwidget)
        self.smoothButton.setGeometry(QtCore.QRect(580, 260, 121, 30))
        self.smoothButton.setObjectName("smoothButton")
        self.smoothButton.setText("Smooth: 1")
        self.smoothButton.clicked.connect(self.pressSmoothButton)

        self.sigma_smooth = 1
        self.sliderSmooth = QtWidgets.QSlider(self.centralwidget)
        self.sliderSmooth.setGeometry(QtCore.QRect(580, 290, 121, 30))
        self.sliderSmooth.setOrientation(QtCore.Qt.Horizontal)
        self.sliderSmooth.setMinimum(1)
        self.sliderSmooth.setMaximum(5)
        self.sliderSmooth.setValue(1)
        self.sliderSmooth.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderSmooth.setTickInterval(1)
        self.sliderSmooth.setObjectName("sliderSmooth")
        self.sliderSmooth.valueChanged.connect(self.sliderSmoothValueChange)

        # Statistics elements
        self.statisticsButton = QtWidgets.QPushButton(self.centralwidget)
        self.statisticsButton.setGeometry(QtCore.QRect(580, 20, 121, 30))
        self.statisticsButton.setObjectName("statisticsButton")
        self.statisticsButton.setText("Statistics")
        self.statisticsButton.clicked.connect(self.pressStatisticsButton)

        self.statisticsLabel = QtWidgets.QLabel(self.centralwidget)
        self.statisticsLabel.setGeometry(QtCore.QRect(530, 50, 300, 210))
        self.statisticsLabel.setText("RMSE: \n PRD: \n MMD:")
        self.statisticsLabel.setObjectName("statisticsLabel")
        self.statisticsLabel.setAlignment(QtCore.Qt.AlignLeft)



        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")

        self.menuFile = QtWidgets.QMenu(self.menubar) # FILE
        self.menuFile.setObjectName("menuFile")  # FILE

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())   # FILE
        self.menuFile.addAction(self.actionLoad_npy)   # FILE



        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.generateButton.setText(_translate("MainWindow", "Generate"))
        self.verifyButton.setText(_translate("MainWindow", "Verify"))

        self.actionLoad_npy.setText(QtCore.QCoreApplication.translate("MainWindow", u"Load npy", None))
        self.menuFile.setTitle(QtCore.QCoreApplication.translate("MainWindow", u"File", None))

    def loadNPY(self):
        name = QtWidgets.QFileDialog.getOpenFileName(None, "Open File")#, os.getenv('HOME'))
        arr = np.load(name[0])

    def numFieldEdited(self):
        if self.generateNumField.text().isdigit():
            num = int(self.generateNumField.text())
            self.amtToGenerate = num
            print(self.amtToGenerate)
            self.generateManyLabel.setText("# of signals to generate: " + str(self.amtToGenerate))
        else:
            print("please enter integer")

    """
    Generate a certain number of samples based on the amount specified
    """
    def pressGenerateManyButton(self):
        batch = []
        # Generate amtToGenerate # of signals
        for iter in range(self.amtToGenerate):
            # Generate a base signal
            seed = tf.random.normal([1, 47, 1])
            ecg = self.generator_model(seed, training=False)
            ecg = ecg.numpy()[0, 0, :]
            ecg = ecg[:187] * self.gen_norm_value
            # Normalize values betwen [0-1] since that's what the encoder model expects
            ecg = sklearn.preprocessing.minmax_scale(ecg, feature_range=(0, 1), axis=0, copy=True)

            for operation in self.operations: # Apply operations to said signal
                if operation.typename == "inc/dec":
                    magnitude = operation.val
                    indices = operation.extras
                    for index in indices:
                        ecg[index] += magnitude
                elif operation.typename == "smooth":
                    sigma_magnitude = operation.val
                    ecg = scipy.ndimage.gaussian_filter1d(ecg, sigma_magnitude)
                else:
                    print("No operations found, generated " + str(self.amtToGenerate) + " # of signals")
            batch.append(ecg)
        batch = np.array(batch)
        np.save("./batchGeneration/batch.npy", batch)





    """
    Call the Generator and generate a signal from random noise, plot the signal
    """
    def pressGenerateButton(self):
        print("Pressed Generate")
        self.operations = [] # Clear current operations slate
        seed = tf.random.normal([1, 47, 1])
        ecg = self.generator_model(seed, training=False)
        ecg = ecg.numpy()[0,0,:]
        ecg = ecg[:187] * self.gen_norm_value
        # Normalize values betwen [0-1] since that's what the encoder model expects
        ecg = sklearn.preprocessing.minmax_scale(ecg, feature_range=(0, 1), axis=0, copy=True)
        self.y_values = ecg # Send values to global
        self.plotTitle = "Generated ECG Signal"
        self.plotPoints()

    """
    Call the Encoder on the current signal state, and then verify the signal with the Generator,
    and then plot the verified signal
    """
    def pressVerifyButton(self):
        print("Pressed Verify")

        # TESTING AE CODE
        # Inference Example loading directly from csv
        test_values = self.y_values

        test_values = np.expand_dims(test_values, axis=0)
        test_values = torch.from_numpy(test_values).float()
        encode_out = self.autoEncoder_model.Encoder(test_values)
        print("Encoded")
        encode_1 = encode_out[0].detach().numpy() # (1,20)
        encode_2 = encode_out[1].detach().numpy() # (1,20)
        stack = np.append(encode_1, encode_2, axis=1) # (1,40)
        stack = np.append(stack, encode_1[:,:7], axis=1) # (1,47) from first 7 rows of encode_1
        stack = np.expand_dims(stack, axis=2) # (1,47,1)
        stack = stack / np.linalg.norm(stack) # normalize because that's what the generator expects

        # Generate new ECG
        verify_ecg = self.generator_model(tf.convert_to_tensor(stack), training=False)
        verify_ecg = verify_ecg.numpy()[0, 0, :]
        verify_ecg = verify_ecg[:187] * self.gen_norm_value
        verify_ecg = sklearn.preprocessing.minmax_scale(verify_ecg, feature_range=(0, 1), axis=0, copy=True)
        self.y_values = verify_ecg
        self.plotTitle = "Verified ECG Signal"
        self.plotPoints()
        # print(stack.shape)


        # # Decode
        # decode_out = self.autoEncoder_model.Decoder(encode_out[0], encode_out[1])
        # # print("Decoded shape before flip: " + str(decode_out.shape))
        #
        # # Flip signal and stuff
        # decode_out = torch.flip(decode_out, dims=[1]), torch.log(F.softmax(encode_out[0], dim=1))
        # decode_out_forplot = decode_out[0].detach().numpy().flatten() # Convert to a numpy array
        # self.y_values = decode_out_forplot # Set global for plotting

        # self.plotPoints()

    def slider1ValueChange(self):
        val = self.slider1.value() # Grab the slider value
        self.sliderLabel1.setText("Element selected: " + str(val)) # Set the slider label text
        self.currentlySelectedPoint = val # Set currently selected point
        self.sliderSpreadValueChange()

    def slider1Released(self):
        self.sliderSpreadValueChange() # Update spread values if we move which point is being selected
        self.plotPoints()

    def sliderSpreadValueChange(self):
        # Clamp to points >= 0 or <= 186
        self.spreadPoints = [i for i in range(self.currentlySelectedPoint - self.sliderSpread.value(), self.currentlySelectedPoint + self.sliderSpread.value() + 1) if i >= 0 and i <= 186]
        # print(self.spreadPoints)
        self.sliderSpreadLabel.setText("Spread: " + str(self.sliderSpread.value()))
        self.plotPoints()

    def pressUpAdjustButton(self):
        for point in self.spreadPoints:
            self.y_values[point] += self.adjustAmt
        # Add operation to operations list for more generation
        if self.adjustAmt > 0:
            op = Operation("inc/dec", self.adjustAmt, self.spreadPoints)
            self.operations.append(op)
        self.plotPoints()

    def pressDownAdjustButton(self):
        for point in self.spreadPoints:
            self.y_values[point] -= self.adjustAmt
        # Add operation to operations list for more generation
        if self.adjustAmt > 0:
            op = Operation("inc/dec", -1 * self.adjustAmt, self.spreadPoints)
            self.operations.append(op)
        self.plotPoints()

    """
    Adjust slider changes the amount to increment/decrement by
    """
    def sliderAdjustValueChange(self):
        self.adjustAmt = self.sliderAdjust.value() * 0.1
        self.sliderAdjustLabel.setText("Inc/Dec Amount: " + str(self.adjustAmt))

    def pressSmoothButton(self):
        self.y_values = scipy.ndimage.gaussian_filter1d(self.y_values, self.sigma_smooth)
        op = Operation("smooth", self.sigma_smooth, [])
        self.operations.append(op)
        self.plotPoints()

    def sliderSmoothValueChange(self):
        self.sigma_smooth = self.sliderSmooth.value()
        self.smoothButton.setText("Smooth: " + str(self.sigma_smooth))

    # Statistics Tests: https://github.com/MikhailMurashov/ecgGAN
    def pressStatisticsButton(self):
        mmd_sum, prd_sum, rmse_sum = [], [], []
        real_data = pd.read_csv("testingData/normal_valid.csv").iloc[:,:187]
        real_data = np.array(real_data)
        for i in range(real_data.shape[0]):
            real_ecg = real_data[i]
            prd_sum.append(self.prd(real_ecg, self.y_values))
            rmse_sum.append(self.rmse(real_ecg, self.y_values))
            mmd_sum.append(self.mmd(real_ecg, self.y_values))
        mmd_str = 'MMD:' + '\n' + f'mean={np.mean(mmd_sum):.4f}' + '\n' + f'min={np.min(mmd_sum):.4f}' + '\n' + f'max={np.max(mmd_sum):.4f}'
        prd_str = 'PRD:' + '\n' + f'mean={np.mean(prd_sum):.4f}' + '\n' + f'min={np.min(prd_sum):.4f}' + '\n' + f'max={np.max(prd_sum):.4f}'
        rmse_str = 'RMSE:' + '\n' + f'mean={np.mean(rmse_sum):.4f}' + '\n' + f'min={np.min(rmse_sum):.4f}' + '\n' + f'max={np.max(rmse_sum):.4f}'
        self.statisticsLabel.setText(rmse_str + "\n" + prd_str + "\n" + mmd_str)

    def rmse(self, targets, predictions):
        return np.sqrt(np.mean((targets - predictions) ** 2))

    def prd(self, targets, predictions): # percent root mean square difference
        s1 = np.sum((targets - predictions) ** 2)
        s2 = np.sum(targets ** 2)
        return np.sqrt(s1 / s2 * 100)

    def mmd(self, targets, predictions):
        mmd_stat = MMDStatistic(187, 187)
        sample_target = torch.from_numpy(targets.reshape((187, 1)))
        sample_pred = torch.from_numpy(predictions.reshape((187, 1)))

        stat = mmd_stat(sample_target, sample_pred, [1.])
        return (stat.item())


    """
    Take global x, y plot values, create and save a plt plot and show the plot on the plotLabel
    """
    def plotPoints(self):
        color_arr = [(0,0,1)] * 187 # Color all points in blue first
        for point in self.spreadPoints:
            color_arr[point] = (1,0,0) # Color selected point in red
        plt.scatter(self.x_values, self.y_values, c = color_arr)
        plt.plot(self.x_values, self.y_values)
        plt.title(self.plotTitle)
        plt.savefig("generatedPlot.png")
        plt.close()
        self.plotLabel.setPixmap(QtGui.QPixmap("generatedPlot.png"))

def except_hook(cls, exception, traceback): # Restore errors, DELETE LATER?
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    sys.excepthook = except_hook # Restore errors, DELETE LATER?

    sys.exit(app.exec_())
