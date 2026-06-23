# Artificial Bee Colony Long Short-Term Memory
### Stock Price Prediction Using Deep LSTM Network Optimized with Artificial Bee Colony Algorithm

## Code

### Install dependencies

The code was tested with **Python 3.13**.

You can install all required dependencies using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

###  Train and Test

1、 Train model

```
python train.py --symbol <symbol of company> --metaheuristic <metaheuristic algorithm> --save-dir <directory to save best parameters>
```

2、 Test model

```
python test.py --symbol <symbol of company> --metaheuristic <metaheuristic algorithm> --load-dir <directory to load parameters>
```

## License

MIT LICENSE
Copyright © 2025 Qui Hoang, Chau Nguyen, Nghia Tran
