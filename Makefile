build:
	docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t mf_kmc -f Dockerfile .
	#	docker build -t mf_kmc .
	# build others images
	docker-compose build

up:
		docker-compose up -d

down:
		docker-compose down

up-dev:
		docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

down-dev:
		docker-compose -f docker-compose.yml -f docker-compose.dev.yml down -v

.PHONY: build build-mocks up down up-dev down-dev
