# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main-window.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 489)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.generateButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateButton.setGeometry(QtCore.QRect(530, 370, 121, 51))
        self.generateButton.setObjectName("generateButton")
        self.generateButton.clicked.connect(self.pressGenerateButton)

        self.plotLabel = QtWidgets.QLabel(self.centralwidget)
        self.plotLabel.setGeometry(QtCore.QRect(10, 10, 511, 411))
        self.plotLabel.setText("")
        self.plotLabel.setPixmap(QtGui.QPixmap("test.png"))
        self.plotLabel.setObjectName("plotLabel")

        self.verifyButton = QtWidgets.QPushButton(self.centralwidget)
        self.verifyButton.setGeometry(QtCore.QRect(660, 370, 121, 51))
        self.verifyButton.setObjectName("verifyButton")


        # Slider and value
        self.sliderLabel = QtWidgets.QLabel(self.centralwidget)
        self.sliderLabel.setGeometry(QtCore.QRect(580, 320, 160, 22))
        self.sliderLabel.setText("Value")
        # self.sliderLabel.setAlignment(QtCore.AlignCenter)
        self.sliderLabel.setObjectName("sliderLabel")

        self.slider1 = QtWidgets.QSlider(self.centralwidget)
        self.slider1.setGeometry(QtCore.QRect(580, 340, 160, 22))
        self.slider1.setOrientation(QtCore.Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(20)
        self.slider1.setValue(0)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider1.setTickInterval(5)
        self.slider1.setObjectName("slider1")
        # self.slider1.valueChanged.connect(self.slider1ValueChange) # Update label when value changed
        self.slider1.sliderReleased.connect(self.slider1ValueChange) # Update label only when slider released

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

    def pressGenerateButton(self):
        self.plotLabel.setPixmap(QtGui.QPixmap("test2.jpg"))

    def slider1ValueChange(self):
        val = self.slider1.value()
        self.sliderLabel.setText(str(val))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
