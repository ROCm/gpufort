DIRECTORIES = $(shell find * -maxdepth 1 -type d)

.PHONY: clean

clean:
	for d in $(DIRECTORIES); do make -C $$d clean; done
