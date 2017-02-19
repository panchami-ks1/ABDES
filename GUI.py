import os
import wx
import wx.lib.agw.multidirdialog as MDD

from Evaluation import diagramEvaluationNew

wildcard = "Python source (*.py)|*.py|" \
           "All files (*.*)|*.*"


########################################################################
class MyForm(wx.Frame):
    # ----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY,
                          "Adaptive Block Diagram Evaluation System ")
        self.panel = wx.Panel(self, wx.ID_ANY)

        self.currentDirectory = os.getcwd()

        self.selected_file = ""
        self.label_1 = wx.StaticText(self.panel, wx.ID_ANY, ("ABDES"))
        self.label_2 = wx.StaticText(self.panel, wx.ID_ANY, (""))

        # create the buttons and bindings
        openFileDlgBtn = wx.Button(self.panel, label="Browse image file")
        openFileDlgBtn.Bind(wx.EVT_BUTTON, self.onOpenFile)

        submit_button = wx.Button(self.panel, label="Submit")
        submit_button.Bind(wx.EVT_BUTTON, self.testMethord)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.label_1, 0, wx.ALIGN_CENTER_HORIZONTAL, 0)
        self.sizer.Add(openFileDlgBtn, 0, wx.ALL | wx.CENTER, 5)
        self.sizer.Add(submit_button, 0, wx.ALL | wx.CENTER, 5)
        self.sizer.Add(self.label_2, 0, wx.CENTER | wx.ALL, 10)
        self.panel.SetSizer(self.sizer)


    # ----------------------------------------------------------------------
    def onOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory,
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            print "You chose the following file(s):"
            for path in paths:
                print path
                self.selected_file = path
        dlg.Destroy()

    def testMethord(self, event):
        complete_file_path = str(self.selected_file)
        if complete_file_path == "":
            print "You havn't selected any files."
            self.label_2.SetLabel("You havn't selected any files.")
            self.sizer.Layout()

        elif complete_file_path.endswith(".png") or complete_file_path.endswith(".jpg") or complete_file_path.endswith(".bmp"):
            file_name = str(os.path.basename(complete_file_path))
            print file_name  + "diag1.jpg"
            print "diag1.jpg"
            score = diagramEvaluationNew(file_name)
            self.label_2.SetLabel("Calculated Score : " + str(int(score)))
            self.sizer.Layout()
            self.selected_file = ""

        else:
            print "Invalid file. Please select an image file."
            self.label_2.SetLabel("Invalid file. Please select an image file.")
            self.sizer.Layout()

# ----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()

