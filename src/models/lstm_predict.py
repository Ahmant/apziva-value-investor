from datetime import datetime, timedelta
import numpy as np

def generate_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    # start_date = start_date_str
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    # end_date = end_date_str

    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return date_list


def predict_date(model, data, data_last_date, date_to_predict, window_size):
    predictions = []

    dates = generate_dates(data_last_date, date_to_predict)[1:]

    X = [data[i] for i in range(window_size)]

    for date in dates:
        predicted_value = float(model.predict([X]).flatten()[0])
        predictions.append(predicted_value)

        # Remove first element
        X.pop(0)

        # Add the new prediction, so we can predict the next date
        X.append(predictions[-1])

    return dates, np.array(predictions).flatten()
