import pandas as pd

mapper = pd.ExcelFile('./resources/mapper.xlsx')
categories = ['Ballad', 'Rock', 'Pop', 'Hiphop', 'Trot', 'Dance', 'RnB']
tags = ['husky', 'clean', 'soulful', 'powerful', 'heavy', 'light', 'angang', 'calm', 'exciting', 'still', 'sad', 'sweet', 'groovy', 'drowsy', 'dreamy']

for category in categories:
    DF = pd.read_excel(mapper, category)
    count = 0
    errors = []
    for i in range(len(DF)):
        _tags = DF.iloc[i, 3:].values
        recordedTags = [tag for tag in _tags if pd.notnull(tag)]
        for recordedTag in recordedTags:
            if recordedTag.lower() not in tags:
                count += 1
                errors.append(recordedTag)
    print(f"{category}: {count}")
    print(errors)