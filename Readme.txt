Project structure:
the project contains two python files pricePrediction.py and utils.py.
Also, there is a file "price_predictions.txt" which contains prices of companies' stocks predicted for 4'th of April.

Requirements for running:
tensorflow 2.8.0
pandas 1.3.4
sklearn 1.0.1

How to run:
python pricePrediction.py <csv file>

Overview of implementation:
The main model for prediction is based on recurrent neural networks, specifically the LSTM networks were used.
The prices data upon which the model was trained was organized as follows:
    every instance was constructed using sliding windows approach,
    specifically the window of size 30 was used, which means that the 1st instance contains the price for the first 30 days
    from 1st day to 30th day.
    the second instance contains the prices from 2nd day to 31 day and etc.
This window size upon which the data was constructed is parametrized, so several values were tried and the best one was chosen.