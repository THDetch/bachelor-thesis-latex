all: compile clean

compile:
	latexmk -pdf main.tex
clean:
	latexmk -c

