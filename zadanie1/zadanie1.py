# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import KFold

# %%
weather_df = pd.read_csv("weather_data.csv")


weather_df['UV Index'] = pd.to_numeric(weather_df['UV Index'], errors='coerce')
print(weather_df.info())

# %%
#detekcia a odstranenie outlierov kod z cvicenia
def detectOutliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outlier_columns = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure','UV Index', 'Visibility (km)']

for column in outlier_columns:
    outliers = detectOutliers(weather_df, column)
    weather_df = weather_df.drop(outliers.index)
    
    plt.figure(figsize=(10,6))

    plt.scatter(weather_df.index, weather_df[column], color='blue', label='Non-Outliers', alpha=0.5)
    plt.scatter(outliers.index, outliers[column], color='red', label='Outliers', s=50)

    plt.xlabel(column)
    plt.ylabel("Values")
    plt.legend()
    plt.show()



# %%
nullValues = weather_df.isnull().sum()
duplicates = weather_df.duplicated().sum()
samples = weather_df.shape[0]

print(nullValues)
print("Null values total: ",nullValues.sum())
print("Duplicates:", duplicates)
print("Samples: ", samples)

# %%
#odstranenie stlpcov v ktorych je vela null hodnot, duplicit a null hodnot
weather_df_cleaned = weather_df.drop('Irradiance', axis=1)
weather_df_cleaned = weather_df_cleaned.dropna()
weather_df_cleaned = weather_df_cleaned.drop_duplicates()

nullValues = weather_df_cleaned.isnull().sum()
duplicates = weather_df_cleaned.duplicated().sum()
samples = weather_df_cleaned.shape[0]

print(nullValues)
print("Null values total: ",nullValues.sum())
print("Duplicates:", duplicates)
print("Samples: ", samples)


# %%

weather_df.reset_index(drop=True, inplace=True)
one_hot_encoded_columns = ['Cloud Cover', 'Season', 'Location']
weather_df_cleaned_encoded = weather_df_cleaned.copy()
for column in one_hot_encoded_columns:
    one_hot_encoded = pd.get_dummies(weather_df_cleaned_encoded[column])
    weather_df_cleaned_encoded = pd.concat([weather_df_cleaned_encoded.drop(columns=column), one_hot_encoded.astype(int)], axis = 1)

label_encoder = LabelEncoder()
weather_df_cleaned_encoded['Weather Type'] = label_encoder.fit_transform(weather_df_cleaned_encoded['Weather Type'])




# %%
X = weather_df_cleaned_encoded.drop('Weather Type', axis=1)
Y = weather_df_cleaned_encoded['Weather Type']

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size = 0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp , Y_temp, test_size = 0.5, random_state=42)


X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape

# %%
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)



# %%
#logisticRegression
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

accurracy = accuracy_score(Y_test, y_pred)
print(f"model accuracy: {accurracy}")

conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'], yticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
plt.title("Konfuzna matica Testovacie data")
plt.xlabel("Predikovane")
plt.ylabel("Aktualne")


conf_matrix = confusion_matrix(Y_train, y_pred_train)
plt.subplot(1,2,2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'], yticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
plt.title("Konfuzna matica Trenovacie data")
plt.xlabel("Predikovane")
plt.ylabel("Aktualne")
plt.show()

# %%
#DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

accurracy = accuracy_score(Y_test, y_pred)
print(f"model accuracy: {accurracy}")

conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'], yticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
plt.title("Konfuzna matica Testovacie data")
plt.xlabel("Predikovane")
plt.ylabel("Aktualne")


conf_matrix = confusion_matrix(Y_train, y_pred_train)
plt.subplot(1,2,2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'], yticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
plt.title("Konfuzna matica Trenovacie data")
plt.xlabel("Predikovane")
plt.ylabel("Aktualne")
plt.show()

# %%
#KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

accurracy = accuracy_score(Y_test, y_pred)
print(f"model accuracy: {accurracy}")

conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'], yticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
plt.title("Konfuzna matica Testovacie data")
plt.xlabel("Predikovane")
plt.ylabel("Aktualne")


conf_matrix = confusion_matrix(Y_train, y_pred_train)
plt.subplot(1,2,2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'], yticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
plt.title("Konfuzna matica Trenovacie data")
plt.xlabel("Predikovane")
plt.ylabel("Aktualne")
plt.show()

# %%
correlation_matrix = weather_df_cleaned.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='crest', linewidths=0.5).set(title="Correlation Matrix")
plt.show()

# %%
temperature_by_season = weather_df_cleaned.groupby('Cloud Cover')['Temperature'].mean().reset_index()

plt.figure(figsize=(10, 8))
sns.barplot(temperature_by_season, x='Cloud Cover', y='Temperature', palette="crest").set(title="Average temperature by cloud cover", ylabel="Average Temperature")
plt.show()

# %%
plt.figure(figsize=(10, 8))
sns.lineplot(weather_df_cleaned, x='Precipitation (%)', y='Humidity').set(title="Precipitation (%) vs Humidity")
plt.show()

# %%
plt.figure(figsize=(10, 8))
sns.displot(weather_df_cleaned, x='UV Index', hue="Weather Type", kind="kde").set(title="Distribution of UV Index values by Weather Type")
plt.show()

# %%
plt.figure(figsize=(12, 8))
sns.scatterplot(weather_df_cleaned, x='Temperature', y='Atmospheric Pressure', hue="Weather Type").set(title="Temperature vs Atmospheric Pressure with Weather Type Classification")
plt.show()

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(150, input_shape=(19,), activation='relu'))
model.add(tf.keras.layers.Dense(70, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, Y_train, epochs=400, batch_size=128, verbose=1, validation_data=(X_val, Y_val), callbacks=[early_stopping])

loss, accurracy = model.evaluate(X_test, Y_test)
train_loss, train_accurracy = model.evaluate(X_train, Y_train)
print(f'Test accuracy: {accurracy:.4f}')
print(f'Training accuracy: {train_accurracy:.4f}')

# %%
plt.figure(figsize=(20, 8))
plt.subplot(1,2,1)
sns.lineplot(x=range(len(history.history['loss'])), y=history.history['loss'], label="Train loss")
sns.lineplot(x=range(len(history.history['val_loss'])), y=history.history['val_loss'], label="Validation loss")
plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
sns.lineplot(x=range(len(history.history['accuracy'])), y=history.history['accuracy'], label="Train accuracy")
sns.lineplot(x=range(len(history.history['val_accuracy'])), y=history.history['val_accuracy'], label="Validation accuracy")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# %%

y_pred = np.argmax(model.predict(X_test), axis=1) 
#Y_test_conf =np.argmax(Y_test, axis=1)
conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'], yticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
plt.title("Konfuzna matica Testovacie data")
plt.xlabel("Predikovane")
plt.ylabel("Aktualne")


y_pred = np.argmax(model.predict(X_train), axis=1) 
#Y_train_conf =np.argmax(Y_train, axis=1)
conf_matrix = confusion_matrix(Y_train, y_pred)
plt.subplot(1,2,2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'], yticklabels=['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
plt.title("Konfuzna matica Trenovacie data")
plt.xlabel("Predikovane")
plt.ylabel("Aktualne")
plt.show()

# %%
#N-fold cross validation
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_accuracies = []
train_accuracies = []
X.reset_index(drop=True, inplace=True)
Y.reset_index(drop=True, inplace=True)
X_scaled = scaler.fit_transform(X)
fold = 1
for train_index, val_index in kf.split(X_scaled):
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    Y_train_fold, Y_val_fold = Y[train_index], Y[val_index]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(300, input_shape=(19,), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_fold, Y_train_fold, epochs=400, batch_size=128, verbose=1, validation_data=(X_val_fold, Y_val_fold), callbacks=[early_stopping])

    loss, accurracy = model.evaluate(X_val_fold, Y_val_fold)
    train_loss, train_accurracy = model.evaluate(X_train_fold, Y_train_fold)
    print(f'Validation accuracy: {accurracy:.4f}')
    print(f'Training accuracy: {train_accurracy:.4f}')

    test_accuracies.append(accurracy)
    train_accuracies.append(train_accurracy)

print(f'Average validation accuracy: {np.mean(test_accuracies):.4f}')
print(f'Average training accuracy: {np.mean(train_accuracies):.4f}')


