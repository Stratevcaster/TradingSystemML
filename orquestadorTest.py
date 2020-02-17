'''
Created on Feb 14, 2020

@author: USER
'''



from test import test 

from parameters import  date_now,LOSS,CELL,N_STEPS,N_LAYERS,UNITS,ticker, LOOKUP_STEP

precios=test(LOOKUP_STEP)
print(precios)
    
     
     
''''    
y_test = data["y_test"]
X_test = data["X_test"]
y_pred = model.predict(X_test)
y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_pred[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))

accuracy_score(y_test, y_pred)

        
y_test = data["y_test"]
X_test = data["X_test"]
  
y_pred = model.predict(X_test)
y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
y_pred_new = numpy.append(y_pred, preciosfutuos)
val = -1600 - len(preciosfutuos)
    
plt.plot(y_test[-1600:], c='b')
plt.plot(y_pred_new[val:], c='r')
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend(["Actual Price", "Predicted Price"])
plt.show()
'''