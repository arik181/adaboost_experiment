FILE = hw3
PRINTER=fabc8802bw1

all : copy

tex : $(FILE).tex
	pdflatex $(FILE)
	pdflatex $(FILE)

copy : tex
	cp $(FILE).pdf ~/public_html/

clean : 
	rm -rf *.aux *.log *.out *.toc *.snm *.nav *.bbl *.blg

print :
	latex $(FILE).tex
	dvips $(FILE).dvi
	lpr -P$(PRINTER) $(FILE).ps
