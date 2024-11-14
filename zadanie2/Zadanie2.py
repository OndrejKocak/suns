# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import TargetEncoder
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# %%
flight_df = pd.read_csv("dataset_flights.csv")
flight_df= flight_df.drop(['ID'], axis=1)
print(flight_df.info())

# %%
def detectOutliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outlier_columns = ['duration', 'days_left', 'price']
for column in outlier_columns:
    outliers = detectOutliers(flight_df, column)
    flight_df = flight_df.drop(outliers.index)
    
    plt.figure(figsize=(10,6))

    plt.scatter(flight_df.index, flight_df[column], color='blue', label='Non-Outliers', alpha=0.5)
    plt.scatter(outliers.index, outliers[column], color='red', label='Outliers', s=50)

    plt.xlabel(column)
    plt.ylabel("Values")
    plt.legend()
    plt.show()

# %%
nullValues = flight_df.isnull().sum()
duplicates = flight_df.duplicated().sum()
samples = flight_df.shape[0]

print(nullValues)
print("Null values total: ",nullValues.sum())
print("Duplicates:", duplicates)
print("Samples: ", samples)

# %%
#odstranenie duplicit a null hodnot
flight_df_cleaned = flight_df.dropna()
flight_df_cleaned = flight_df_cleaned.drop_duplicates()

nullValues = flight_df_cleaned.isnull().sum()
duplicates = flight_df_cleaned.duplicated().sum()
samples = flight_df_cleaned.shape[0]

print(nullValues)
print("Null values total: ",nullValues.sum())
print("Duplicates:", duplicates)
print("Samples: ", samples)


# %%
flight_df_cleaned.reset_index(drop=True, inplace=True)
one_hot_encoded_columns = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class']

flight_df_encoded = pd.get_dummies(flight_df_cleaned,columns=one_hot_encoded_columns, drop_first=False)

flight_df_encoded['stops'] = flight_df_encoded['stops'].map({'zero': 0, 'one': 1, 'two_or_more': 2})

flight_df_encoded['flight'] = flight_df_encoded['flight'].map(flight_df_encoded.groupby('flight')['price'].mean())

# %%
flight_df_encoded.reset_index(drop=True, inplace=True)
X = flight_df_encoded.drop('price', axis=1)
Y = flight_df_encoded['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

# %%
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

r2_pred = r2_score(Y_test, y_pred)
rmse_pred = mean_squared_error(Y_test, y_pred, squared=False)
r2_pred_train = r2_score(Y_train, y_pred_train)
rmse_pred_train = mean_squared_error(Y_train, y_pred_train, squared=False)

print("R2 score test data: ", r2_pred)
print("RMSE test data: ", rmse_pred)
print("R2 score train data: ", r2_pred_train)
print("RMSE test data: ", rmse_pred_train)

residuals = Y_test - y_pred
residuals_train = Y_train - y_pred_train

plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály testovacia mnozina")

plt.subplot(1,2,2)
sns.scatterplot(x=y_pred_train, y=residuals_train)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály trenovacia mnozina")
plt.show()

plt.figure(figsize=(20, 10))
#plot_tree(model, filled=True, feature_names=X.columns.tolist(), rounded=True)
plt.title("Decision Tree Regressor")
plt.show()

# %%
model = RandomForestRegressor()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

r2_pred = r2_score(Y_test, y_pred)
rmse_pred = mean_squared_error(Y_test, y_pred, squared=False)
r2_pred_train = r2_score(Y_train, y_pred_train)
rmse_pred_train = mean_squared_error(Y_train, y_pred_train, squared=False)

print("R2 score test data: ", r2_pred)
print("RMSE test data: ", rmse_pred)
print("R2 score train data: ", r2_pred_train)
print("RMSE test data: ", rmse_pred_train)

residuals = Y_test - y_pred
residuals_train = Y_train - y_pred_train

plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály testovacia mnozina")

plt.subplot(1,2,2)
sns.scatterplot(x=y_pred_train, y=residuals_train)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály trenovacia mnozina")
plt.show()


feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

top_features = importance_df.head(11)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features['Importance'], y=top_features['Feature'], palette="crest")
plt.xlabel("Dôležitosť atribútu")
plt.ylabel("Vstupný atribút")
plt.title("Najdôležitejšie atribúty pre RandomForestRegressor")
plt.show()


# %%
model = SVR(kernel='rbf', C=1000)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

r2_pred = r2_score(Y_test, y_pred)
rmse_pred = mean_squared_error(Y_test, y_pred, squared=False)
r2_pred_train = r2_score(Y_train, y_pred_train)
rmse_pred_train = mean_squared_error(Y_train, y_pred_train, squared=False)
print("R2 score test data: ", r2_pred)
print("RMSE test data: ", rmse_pred)
print("R2 score train data: ", r2_pred_train)
print("RMSE test data: ", rmse_pred_train)

residuals = Y_test - y_pred
residuals_train = Y_train - y_pred_train

plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály testovacia mnozina")

plt.subplot(1,2,2)
sns.scatterplot(x=y_pred_train, y=residuals_train)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály trenovacia mnozina")
plt.show()


# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(50, input_shape=(X_train.shape[1],), activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, Y_train, epochs=400, batch_size=128, verbose=1, validation_split=0.2, callbacks=[early_stopping])

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

mse, mae = model.evaluate(X_test, Y_test)
rmse = math.sqrt(mse)
r2_pred = r2_score(Y_test, y_pred)
mse_train, mae_train =model.evaluate(X_train, Y_train)
rmse_train = math.sqrt(mse)
r2_pred_train = r2_score(Y_train, y_pred_train)
print("R2 score test data: ", r2_pred)
print(f'Test RMSE: {rmse}')
print(f'Test MAE: {mae}' )
print("R2 score train data: ", r2_pred_train)
print(f'Training RMSE: {rmse_train}')
print(f'Train MAE: {mae}' )


residuals = Y_test - y_pred.flatten() 
residuals_train = Y_train - y_pred_train.flatten() 

plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
sns.scatterplot(x=y_pred.flatten(), y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály testovacia mnozina")

plt.subplot(1,2,2)
sns.scatterplot(x=y_pred_train.flatten(), y=residuals_train)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály trenovacia mnozina")
plt.show()


# %%
fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111, projection='3d')

x = flight_df_encoded['stops'] 
y = flight_df_encoded['duration'] 
z = flight_df_encoded['class_Economy'] 


scatter = ax.scatter(x, y, z, c=flight_df_encoded['price'], marker='o')
ax.set_xlabel('Stops')
ax.set_ylabel('Duration')
ax.set_zlabel('class_Economy')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label("Cena")

plt.show()

# %%
pca = PCA(n_components=3)
X_scaled = scaler.fit_transform(X)
X_reduced = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_reduced[:,0], X_reduced[:, 1], X_reduced[:, 2], c=flight_df_cleaned['price'], marker='o')
ax.set_xlabel('PCA 0')
ax.set_ylabel('PCA 1')
ax.set_zlabel('PCA 2')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label("Cena")

plt.show()


# %%
correlation_matrix = flight_df_encoded.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='crest', linewidths=0.5).set(title="Correlation Matrix")
plt.show()

correlations_with_price = correlation_matrix['price'].sort_values(ascending=False)
selected_features = correlations_with_price.index[1:5].tolist()

print("Korelácie s cenou (price), zoradené od najvyššej po najnižšiu:")
print(correlations_with_price)

flight_df_encoded.reset_index(drop=True, inplace=True)
X = flight_df_encoded[selected_features]
Y = flight_df_encoded['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

r2_pred = r2_score(Y_test, y_pred)
rmse_pred = mean_squared_error(Y_test, y_pred, squared=False)
r2_pred_train = r2_score(Y_train, y_pred_train)
rmse_pred_train = mean_squared_error(Y_train, y_pred_train, squared=False)

print("R2 score test data: ", r2_pred)
print("RMSE test data: ", rmse_pred)
print("R2 score train data: ", r2_pred_train)
print("RMSE test data: ", rmse_pred_train)

residuals = Y_test - y_pred
residuals_train = Y_train - y_pred_train

plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály testovacia mnozina")

plt.subplot(1,2,2)
sns.scatterplot(x=y_pred_train, y=residuals_train)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály trenovacia mnozina")
plt.show()


# %%
flight_df_encoded.reset_index(drop=True, inplace=True)
X = flight_df_encoded[['class_Business', 'class_Economy', 'duration', 'flight', 'days_left']]
Y = flight_df_encoded['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

r2_pred = r2_score(Y_test, y_pred)
rmse_pred = mean_squared_error(Y_test, y_pred, squared=False)
r2_pred_train = r2_score(Y_train, y_pred_train)
rmse_pred_train = mean_squared_error(Y_train, y_pred_train, squared=False)

print("R2 score test data: ", r2_pred)
print("RMSE test data: ", rmse_pred)
print("R2 score train data: ", r2_pred_train)
print("RMSE test data: ", rmse_pred_train)

residuals = Y_test - y_pred
residuals_train = Y_train - y_pred_train

plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály testovacia mnozina")

plt.subplot(1,2,2)
sns.scatterplot(x=y_pred_train, y=residuals_train)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály trenovacia mnozina")
plt.show()


# %%
pca = PCA(n_components=0.85)
X_scaled = scaler.fit_transform(X)
X_reduced = pca.fit_transform(X_scaled)


X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size = 0.3, random_state=42)


model = RandomForestRegressor()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

r2_pred = r2_score(Y_test, y_pred)
rmse_pred = mean_squared_error(Y_test, y_pred, squared=False)
r2_pred_train = r2_score(Y_train, y_pred_train)
rmse_pred_train = mean_squared_error(Y_train, y_pred_train, squared=False)

print("R2 score test data: ", r2_pred)
print("RMSE test data: ", rmse_pred)
print("R2 score train data: ", r2_pred_train)
print("RMSE test data: ", rmse_pred_train)

residuals = Y_test - y_pred
residuals_train = Y_train - y_pred_train

plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály testovacia mnozina")

plt.subplot(1,2,2)
sns.scatterplot(x=y_pred_train, y=residuals_train)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikované hodnoty")
plt.ylabel("Reziduály")
plt.title("Reziduály trenovacia mnozina")
plt.show()


# %%
temperature_by_season = flight_df_cleaned.groupby('airline')['duration'].mean().reset_index()

plt.figure(figsize=(10, 8))
sns.barplot(temperature_by_season, x='airline', y='duration', palette="crest").set(title="Average flight duration by airline", ylabel="Average Flight Duration")
plt.show()

# %%
plt.figure(figsize=(10, 8))
sns.displot(flight_df_cleaned, x='price', hue="class", kind="kde").set(title="Distribution of Price values by Class")
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=flight_df_encoded, x='stops', y='duration', hue='stops', palette='crest', s=100, alpha=0.7)
plt.title("Number of stops by flight duration")
plt.xlabel("Stops")
plt.ylabel("Duration")
plt.legend(title="Stops", loc='upper left')
plt.show()

# %%
departure_counts = flight_df_cleaned.groupby(['source_city', 'departure_time']).size().unstack(fill_value=0)
departure_percent = departure_counts.div(departure_counts.sum(axis=1), axis=0) * 100 

departure_percent.plot(kind='bar', stacked=True, figsize=(12, 8), cmap="Set2")
plt.title("Percentage representation of flights by departure time for each departure city")
plt.xlabel("Source_city")
plt.ylabel("Percentage")
plt.legend(title="Departure_time", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()

# %%
arrival_time_counts = flight_df_cleaned['arrival_time'].value_counts(normalize=True) * 100  # Prevod na percentá

plt.figure(figsize=(10, 10))
plt.pie(arrival_time_counts, labels=arrival_time_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set2.colors)
plt.title("Percentuálne zastúpenie letov podľa času príchodu")
plt.show()


