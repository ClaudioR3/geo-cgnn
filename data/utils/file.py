import os
import shutil

class FileStorage: 
    """
    This class allows the comunication with the FileSystem.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
    def read(self, file):
        with open(file, 'rb') as f:
            reader = f.read()
        return reader
    
    def write(self, file, body):
        with open(file, 'wb') as f:
            f.write(body)
            
    def objectslist(self, path):
        return [os.path.join(path,file) for file in os.listdir(path)]
    
    def clean(self,path):
        if os.path.exists(path):
            shutil.rmtree(path)