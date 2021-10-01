model.fit(X_train_encoded, Y_train)

Y_hat_train = model.predict(X_train_encoded)
Y_hat_test = model.predict(X_test_encoded)

acc_train = accuracy_score(Y_train, Y_hat_train)
acc_test = accuracy_score(Y_test.str.replace("\\.$", ""), Y_hat_test)

try:
    results # checks if this object exists or not
except NameError:
    results = pd.DataFrame(columns = ["algo", "acc_train", "acc_test"]) # initiates it

model_name = str(model.__class__).split('.')[-1].strip("\"\'>")
results.loc[len(results), 0:3] = [model_name, acc_train, acc_test]

for hp in hypers.keys():
    results.loc[len(results) - 1, hp] = hypers[hp]
