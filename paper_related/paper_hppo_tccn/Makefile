PAPER := main
BUILD := build

.PHONY: all clean watch wordcount

all:
	mkdir -p $(BUILD)
	latexmk -pdf -outdir=$(BUILD) -interaction=nonstopmode $(PAPER).tex

watch:
	mkdir -p $(BUILD)
	latexmk -pdf -pvc -outdir=$(BUILD) -interaction=nonstopmode $(PAPER).tex

clean:
	latexmk -C -outdir=$(BUILD) $(PAPER).tex
	rm -rf $(BUILD)/*.{aux,bbl,blg,log,out,toc,fdb_latexmk,fls,synctex.gz}

wordcount:
	@which texcount > /dev/null || (echo "texcount not installed"; exit 1)
	texcount -inc -sum $(PAPER).tex
