## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#Process)

## General info
Collect PDF training data:
This project utilize computer vision techniques based on heuristics for table decomposition to detect and extract data from PDF. This project has been motivated upon document analysis ideas found in academic papers
	
## Technologies
Keywords extraction with:
* we extract keywords from PDFs using a program code from an existing public GitHub repository:
https://github.com/tabulapdf/tabula
	
## Process
This process contain five setps: 
* (1) Import all libraries, 
* (2) Convert PDF file to txt format and read data, 
* (3) extract regular expressions to extract keywords, 
* (4) Save list of extracted keywords in a DataFrame and 
* (5) Save the results in a DataFrame into a CSV file.
The obtained csv contain statements that describe hypothesis and information about CMs and its environment test.


