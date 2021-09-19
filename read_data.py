'''
authors: saurav.pattnaik & srishti.verma :)
'''

import PyPDF2
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from nltk.tokenize import sent_tokenize
import os

def return_output_string(path):
    output_string = StringIO()
    with open(path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue()


# THIS FUNCTION WILL BE CALLED
def read_pdfs(filename1, filename2, threshold=1000):

    path1 = os.path.join(os.getcwd(), 'Data Files', str(filename1))
    path2 = os.path.join(os.getcwd(), 'Data Files', str(filename2))

    file1 = return_output_string(path1)
    file2 = return_output_string(path2)

    file1 = sent_tokenize(file1)
    file2 = sent_tokenize(file2)

    return file1[5:min(len(file1), threshold)], file2[5:min(len(file2), threshold)]