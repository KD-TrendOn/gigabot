from langchain_community.document_loaders import *


import pandas as pd


with open("empty.txt", "w") as f:
    f.write("")


class DataAbstraction:
    def __init__(self, filename: str):
        self.type = filename.split(".")[-1]
        self.filename = filename
        if self.type == "csv":
            self.sql_like = True
        elif self.type == "xlsx":
            self.sql_like = True
        else:
            self.sql_like = False

    def get_loader(self):
        if self.type == "csv":
            loader = CSVLoader(file_path=self.filename)
        elif self.type == "json":
            loader = JSONLoader(file_path=self.filename, text_content=False)
        elif self.type == "docx":
            loader = Docx2txtLoader(self.filename)
        elif self.type == "xlsx":
            loader = UnstructuredExcelLoader(self.filename, mode="elements")
        elif self.type == "pdf":
            loader = PyPDFLoader(self.filename)
        elif self.type == "txt":
            loader = TextLoader(self.filename)
        elif self.type == "pptx":
            loader = UnstructuredPowerPointLoader(self.filename)
        elif self.type == "ipynb":
            loader = NotebookLoader(self.filename)
        elif self.type == "html":
            loader = BSHTMLLoader(self.filename)
        else:
            loader = TextLoader("empty.txt")
        return loader

    def get_df(self):
        if self.type == "csv":
            df = pd.read_csv(self.filename)
        elif self.type == "xlsx":
            df = pd.read_csv(self.filename)
        return df
