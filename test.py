from sklearn.preprocessing import LabelEncoder

y = ['TCP','UDP']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

print(y)