import re
import pandas as pd

def preprocessor(data):
    pattern = r'\d{1,2}/\d{1,2}(?:/\d{2})?,\s\d{1,2}:\d{2}\s*[aApP][mM]'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Fix narrow no-break space
    df['message_date'] = df['message_date'].str.replace('\u202f', ' ', regex=False)

    # Parse dates
    df['date'] = pd.to_datetime(df['message_date'], dayfirst=True, infer_datetime_format=True, errors='coerce')
    invalid = df['date'].isna().sum()
    print(f'Unparsed timestamps: {invalid} (of {len(df)})')

    # Split user and message
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name exists
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract datetime features
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['only_date'] = df['date'].dt.date
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # period column (hourly ranges for heatmap)
    periods = []
    for hour in df['hour']:
        if hour == 23:
            periods.append(f"{hour}-00")
        else:
            periods.append(f"{hour}-{hour + 1}")
    df['period'] = periods

    # weekday order for better plotting
    df['day_name'] = pd.Categorical(
        df['day_name'],
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True
    )

    return df
