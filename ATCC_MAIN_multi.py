from multiprocessing import Process, Queue, Pipe
from threading import Thread
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class Emitter(QObject, Thread):

    def __init__(self, transport, parent=None):
        QObject.__init__(self,parent)
        Thread.__init__(self)
        self.transport = transport

    def _emit(self, signature, args=None):
        if args:
            self.emit(SIGNAL(signature), args)
        else:
            self.emit(SIGNAL(signature))

    def run(self):
        while True:
            try:
                signature = self.transport.recv()
            except EOFError:
                break
            else:
                self._emit(*signature)

class Form(QDialog):

    def __init__(self, queue, emitter, parent=None):
        super(Form,self).__init__(parent)
        self.data_to_child = queue
        self.emitter = emitter
        self.emitter.daemon = True
        self.emitter.start()
        self.browser = QTextBrowser()
        self.lineedit = QLineEdit('Type text and press <Enter>')
        self.lineedit.selectAll()
        layout = QVBoxLayout()
        layout.addWidget(self.browser)
        layout.addWidget(self.lineedit)
        self.setLayout(layout)
        self.lineedit.setFocus()
        self.setWindowTitle('Upper')
        self.connect(self.lineedit,SIGNAL('returnPressed()'),self.to_child)
        self.connect(self.emitter,SIGNAL('data(PyQt_PyObject)'), self.updateUI)

    def to_child(self):
        self.data_to_child.put(unicode(self.lineedit.text()))
        self.lineedit.clear()

    def updateUI(self, text):
        text = text[0]
        self.browser.append(text)

class ChildProc(Process):

    def __init__(self, transport, queue, daemon=True):
        Process.__init__(self)
        self.daemon = daemon
        self.transport = transport
        self.data_from_mother = queue

    def emit_to_mother(self, signature, args=None):
        signature = (signature, )
        if args:
            signature += (args, )
        self.transport.send(signature)

    def run(self):
        while True:
            text = self.data_from_mother.get()
            self.emit_to_mother('data(PyQt_PyObject)', (text.upper(),))

if __name__ == '__main__':

    app = QApplication(sys.argv)
    mother_pipe, child_pipe = Pipe()
    queue = Queue()
    emitter = Emitter(mother_pipe)
    form = Form(queue, emitter)
    ChildProc(child_pipe, queue).start()
    form.show()
    app.exec_()
