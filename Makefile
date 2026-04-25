.PHONY: install install-box2d test lint format-check docker-test smoke-reinforce smoke-dqn smoke-ppo

install:
	pip install -e ".[dev]"

install-box2d:
	pip install -e ".[dev,box2d]"

test:
	pytest

lint:
	ruff check .

format-check:
	ruff format --check .

docker-test:
	docker build -t rl-gymnasium-test .
	docker run --rm rl-gymnasium-test

smoke-reinforce:
	python train.py --algorithm reinforce --seed 0 --episodes 2 --device cpu --out-dir outputs/smoke/reinforce

smoke-dqn:
	python train.py --algorithm dqn --seed 0 --episodes 2 --device cpu --out-dir outputs/smoke/dqn

smoke-ppo:
	python train.py --algorithm ppo --seed 0 --iterations 1 --device cpu --out-dir outputs/smoke/ppo
