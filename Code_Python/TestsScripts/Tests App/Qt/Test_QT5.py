# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:50:57 2022

@author: Joseph
"""

#### Tests of Qt5 gui functions


# %%

from PyQt5.QtWidgets import QApplication, QWidget

# Only needed for access to command line arguments
import sys

# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
app = QApplication(sys.argv)

# Create a Qt widget, which will be our window.
window = QWidget()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()


# Your application won't reach here until you exit and the event
# loop has stopped.


# %%

from PyQt5.QtWidgets import QApplication, QWidget
import sys

app = QApplication(sys.argv)
# app = QApplication([])

window = QWidget()
window.show()

# %%

import sys
from PyQt5.QtWidgets import QApplication, QPushButton

app = QApplication(sys.argv)

window = QPushButton("Push Me")
window.show()

app.exec()

# %%

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)

window = QMainWindow()
window.show()

# Start the event loop.
app.exec()


# %%

import sys

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        button = QPushButton("Press Me!")

        # Set the central widget of the Window.
        self.setCentralWidget(button)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

# app.exec()


# %%

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        
        self.button_is_checked = True
        
        qbutton = QPushButton("Quit")
        qbutton.setCheckable(True)
        qbutton.clicked.connect(self.the_qbutton_was_clicked)
    
        # Set the Q widget of the Window.
        self.setMenuWidget(qbutton)     
        
        self.button = QPushButton("Press Me!")
        self.button.setCheckable(True)
        self.button.released.connect(self.the_button_was_released)
        self.button.setChecked(self.button_is_checked)

        self.setCentralWidget(self.button)

    def the_button_was_released(self):
        self.button_is_checked = self.button.isChecked()

        print(self.button_is_checked)

    #     button = QPushButton("Press Me!")
    #     button.setCheckable(True)
    #     button.clicked.connect(self.the_button_was_clicked)
    #     button.clicked.connect(self.the_button_was_toggled)
        
    #     # Set the central widget of the Window.
    #     self.setCentralWidget(button)

    # def the_button_was_clicked(self):
    #     print("Clicked!")

    # def the_button_was_toggled(self, checked):
    #     print("Checked?", checked)
    
    def the_qbutton_was_clicked(self):
        # self.app.quit()
        QApplication.quit()
        self.close()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()


# %%


from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget

import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        
        qbutton = QPushButton("Quit")
        qbutton.clicked.connect(self.quit_button)
        self.setMenuWidget(qbutton)    

        self.label = QLabel()
        self.input = QLineEdit()
        
        self.input.textChanged.connect(self.label.setText)
        self.input.cursorPositionChanged.connect(self.appendText)

        layout = QVBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)

        # Set the central widget of the Window.
        self.setCentralWidget(container)
        
    def appendText(self):
        text = self.label.text()
        print(text)
        self.label.setText(text + '0')
        
    def quit_button(self):
        # self.app.quit()
        QApplication.quit()
        self.close()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()


# %%

import sys

# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import (
#     QApplication,
#     QCheckBox,
#     QComboBox,
#     QDateEdit,
#     QDateTimeEdit,
#     QDial,
#     QDoubleSpinBox,
#     QFontComboBox,
#     QLabel,
#     QLCDNumber,
#     QLineEdit,
#     QMainWindow,
#     QProgressBar,
#     QPushButton,
#     QRadioButton,
#     QSlider,
#     QSpinBox,
#     QTimeEdit,
#     QVBoxLayout,
#     QWidget,
#     QButtonGroup,
#     QSpacerItem,
# )

from PyQt5 import QtWidgets as Qtw

# Subclass QMainWindow to customize your application's main window
class MainWindow(Qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.res = ['Nan', 'Nan']

        self.setWindowTitle("Click on buttons")
        
        qbutton = Qtw.QPushButton("Quit")
        qbutton.clicked.connect(self.quit_button)
        self.setMenuWidget(qbutton)
        
        layout = Qtw.QVBoxLayout()  # layout for the central widget
        main_widget = Qtw.QWidget(self)  # central widget
        main_widget.setLayout(layout)
        
        label1 = Qtw.QLabel('Choice 1')
        rbg1 = Qtw.QButtonGroup(main_widget)
        rb11 = Qtw.QRadioButton('a')
        rb12 = Qtw.QRadioButton('b')
        rbg1.addButton(rb11)
        rbg1.addButton(rb12)
        
        label2 = Qtw.QLabel('Choice 2')
        rbg2 = Qtw.QButtonGroup(main_widget)
        rb21 = Qtw.QRadioButton('1')
        rb22 = Qtw.QRadioButton('2')
        rbg2.addButton(rb21)
        rbg2.addButton(rb22)
        
        valb = Qtw.QPushButton('OK', main_widget)
        
        layout.addWidget(label1)
        layout.addWidget(rb11)
        layout.addWidget(rb12)
        layout.addSpacing(20)
        layout.addWidget(label2)
        layout.addWidget(rb21)
        layout.addWidget(rb22)
        layout.addSpacing(20)
        layout.addWidget(valb)
        
        self.rbg1 = rbg1
        self.rbg2 = rbg2
        self.valb = valb
        self.setCentralWidget(main_widget)
        
        valb.clicked.connect(self.validate_button)
        
        
    def quit_button(self):
        Qtw.QApplication.quit()
        self.close()
        
    def validate_button(self):
        err = (self.rbg1.checkedButton() == None) or (self.rbg2.checkedButton() == None)
        s1 = self.rbg1.checkedButton().text()
        s2 = self.rbg2.checkedButton().text()
        self.res = [s1, s2]
        Qtw.QApplication.quit()
        self.close()


def main():
    app = Qtw.QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
        
    app.exec()
    res = window.res
    print(res)
    

main()


# %% Works well !

import sys
import numpy as np
from PyQt5 import QtWidgets as Qtw


class ChoicesBox(Qtw.QMainWindow):
    def __init__(self, choicesDict, title = 'Multiple choice box'):
        super().__init__()
        
        
        self.choicesDict = choicesDict
        self.questions = [k for k in choicesDict.keys()]
        self.nQ = len(self.questions)
        
        self.res = {}
        self.list_rbg = [] # rbg = radio button group

        self.setWindowTitle(title)
        
        layout = Qtw.QVBoxLayout()  # layout for the central widget
        main_widget = Qtw.QWidget(self)  # central widget
        main_widget.setLayout(layout)
        
        for q in self.questions:
            choices = self.choicesDict[q]
            label = Qtw.QLabel(q)
            layout.addWidget(label)
            rbg = Qtw.QButtonGroup(main_widget)
            for c in choices:
                rb = Qtw.QRadioButton(c)
                rbg.addButton(rb)
                layout.addWidget(rb)
                
            self.list_rbg.append(rbg)
            layout.addSpacing(20)
        
        valid_button = Qtw.QPushButton('OK', main_widget)
        layout.addWidget(valid_button)
        
        self.setCentralWidget(main_widget)
        
        valid_button.clicked.connect(self.validate_button)


    def validate_button(self):
        array_err = np.array([rbg.checkedButton() == None for rbg in self.list_rbg])
        Err = np.any(array_err)
        if Err:
            self.error_dialog()
        else:
            for i in range(self.nQ):
                q = self.questions[i]
                rbg = self.list_rbg[i]
                self.res[q] = rbg.checkedButton().text()
                
            self.quit_button()
            
    def error_dialog(self):
        dlg = Qtw.QMessageBox(self)
        dlg.setWindowTitle("Error")
        dlg.setText("Please make a choice in each category.")
        dlg.exec()
        
    def quit_button(self):
        Qtw.QApplication.quit()
        self.close()


def main(choicesDict):
    app = Qtw.QApplication(sys.argv)
    
    box = ChoicesBox(choicesDict)
    box.show()
        
    app.exec()
    res = box.res
    return(res)

choicesDict = {'Is the cell ok?' : ['Yes', 'No'],
               'Is the nucleus visible?' : ['Yes', 'No'],}
res = main(choicesDict)


