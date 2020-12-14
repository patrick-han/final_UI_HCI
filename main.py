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

# Encoder imports
from mymodel import LSTMEncoder
from mymodel import AE_opt
import torch
import torch.optim as optim
import torch.nn.functional as F


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 489)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Generator run-through
        self.generateButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateButton.setGeometry(QtCore.QRect(530, 370, 121, 51))
        self.generateButton.setObjectName("generateButton")
        self.generateButton.clicked.connect(self.pressGenerateButton)

        # Plot area elements
        self.plotLabel = QtWidgets.QLabel(self.centralwidget)
        self.plotLabel.setGeometry(QtCore.QRect(10, 10, 511, 411))
        self.plotLabel.setText("")
        self.plotLabel.setPixmap(QtGui.QPixmap("test.png"))
        self.plotLabel.setScaledContents(True) # Fit to label
        self.plotLabel.setObjectName("plotLabel")

        # Plot variables
        df = pd.read_csv("testingData/normal_valid.csv")
        arr = df.iloc[0][:-1]
        self.x_values = np.linspace(0, arr.shape[0],arr.shape[0])
        self.y_values = arr


        # Encoder run-through
        self.verifyButton = QtWidgets.QPushButton(self.centralwidget)
        self.verifyButton.setGeometry(QtCore.QRect(660, 370, 121, 51))
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
        self.sliderLabel1.setGeometry(QtCore.QRect(530, 320, 160, 22))
        self.sliderLabel1.setText("Element Selected: 0")
        self.sliderLabel1.setObjectName("sliderLabel")

        self.slider1 = QtWidgets.QSlider(self.centralwidget)
        self.slider1.setGeometry(QtCore.QRect(530, 340, 250, 22))
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
        self.sliderSpreadLabel.setGeometry(QtCore.QRect(530, 270, 160, 22))
        self.sliderSpreadLabel.setText("Spread: 0")
        self.sliderSpreadLabel.setObjectName("sliderSpreadLabel")

        self.sliderSpread = QtWidgets.QSlider(self.centralwidget)
        self.sliderSpread.setGeometry(QtCore.QRect(530, 290, 250, 22))
        self.sliderSpread.setOrientation(QtCore.Qt.Horizontal)
        self.sliderSpread.setMinimum(0)
        self.sliderSpread.setMaximum(5)  # Select up to a spread of 5 on each side
        self.sliderSpread.setValue(0)
        self.sliderSpread.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderSpread.setTickInterval(1)
        self.sliderSpread.setObjectName("sliderSpread")
        self.sliderSpread.valueChanged.connect(self.sliderSpreadValueChange)  # Update label when value changed
        # self.sliderSpread.sliderReleased.connect(self.sliderSpread1Released)  # Update label only when slider released

        self.currentlySelectedPoint = 0 # The currently selected point will be highlighted in red on the scatter
        self.spreadPoints = [0]

        # Buttons that allow you to adjust the height of the selected point
        self.upAdjustButton = QtWidgets.QPushButton(self.centralwidget)
        self.upAdjustButton.setGeometry(QtCore.QRect(530, 30, 30, 30))
        self.upAdjustButton.setObjectName("upAdjustButton")
        self.upAdjustButton.setText("Inc.")
        self.upAdjustButton.clicked.connect(self.pressUpAdjustButton)

        self.downAdjustButton = QtWidgets.QPushButton(self.centralwidget)
        self.downAdjustButton.setGeometry(QtCore.QRect(530, 70, 30, 30))
        self.downAdjustButton.setObjectName("downAdjustButton")
        self.downAdjustButton.setText("Dec.")
        self.downAdjustButton.clicked.connect(self.pressDownAdjustButton)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.generateButton.setText(_translate("MainWindow", "Generate"))
        self.verifyButton.setText(_translate("MainWindow", "Verify"))

    """
    Call the Generator and generate a signal, plot the signal
    """
    def pressGenerateButton(self):
        print("Pressed Generate")
        # TODO: Call generator function

        # TODO: Plot and display generator points
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
        # print("Encoded")

        # Decode
        decode_out = self.autoEncoder_model.Decoder(encode_out[0], encode_out[1])
        # print("Decoded shape before flip: " + str(decode_out.shape))

        # Flip signal and stuff
        decode_out = torch.flip(decode_out, dims=[1]), torch.log(F.softmax(encode_out[0], dim=1))
        decode_out_forplot = decode_out[0].detach().numpy().flatten() # Convert to a numpy array
        self.y_values = decode_out_forplot # Set global for plotting
        # self.plotPoints()

        ##DEELELTLE
        # plt.plot(self.x_values, self.y_values)
        # plt.plot(self.x_values, decode_out_forplot)
        # plt.savefig("generatedPlot.png")
        # plt.close()
        # self.plotLabel.setPixmap(QtGui.QPixmap("generatedPlot.png"))

        # # Plot original
        # plt.plot(example.numpy().flatten())
        # # Plot decoded
        # plt.plot(decode_out[0].detach().numpy()[0])

        # TODO: Call encoder on current signal
        # encoderOutput = Encoder()
        # TODO: Call generator on encoderOutput, modify global x, y values
        # plotPoints()

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
            self.y_values[point] += 0.1
        self.plotPoints()

    def pressDownAdjustButton(self):
        for point in self.spreadPoints:
            self.y_values[point] -= 0.1
        self.plotPoints()

    """
    Take global x, y plot values, create and save a plt plot and show the plot on the plotLabel
    """
    def plotPoints(self):
        color_arr = [(0,0,1)] * 187 # Color all points in blue first
        for point in self.spreadPoints:
            color_arr[point] = (1,0,0) # Color selected point in red
        plt.scatter(self.x_values, self.y_values, c = color_arr)
        plt.plot(self.x_values, self.y_values)
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
