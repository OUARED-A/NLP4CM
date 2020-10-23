Collect PDF training data:

we extract keywords from PDFs using a program code from an existing public GitHub repository:
ttps://github.com/tabulapdf/tabula

This project utilize computer vision techniques based on heuristics for table decomposition to detect and extract data from PDF. This project has been motivated upon document analysis ideas found in academic papers by [1], [2]. This process contain five setps: (1) Import all libraries, (2) Convert PDF file to txt format and read data, (3)
extract regular expressions to extract keywords, (4) Save list of extracted keywords in a DataFrame and (5) Save the results in a DataFrame into a CSV file. The obtained
csv contain statements that describe hypothesis and information about CMs and its environment test.

1. T. Hassan and R. Baumgartner. Table recognition and understanding from pdf
files. In Ninth International Conference on Document Analysis and Recognition
(ICDAR 2007), volume 2, pages 1143{1147. IEEE, 2007.

2. B. Yildiz, K. Kaiser, and S. Miksch. pdf2table: A method to extract table information from pdf files. In IICAI, pages 1773{1785, 2005.
