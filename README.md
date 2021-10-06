# car-music

to run application using docker do:
```bash
docker pull lechwolowski/car-music

docker run -it lechwolowski/car-music
```

to run locally do:

```bash
pip install -r requirements.txt
```

#### Create one-hot encoded dataset version
```bash
python -m dataset.data_preparation
```

```bash
python -m main.py
```

or run scripts independently:
```bash
python -m train.py
python -m inference.py
```

to test do:
```bash
export PYTHONPATH=`pwd`
pip install -r test-requirements.txt
```
than:
```bash
pytest
```