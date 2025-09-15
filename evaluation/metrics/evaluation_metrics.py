def accuracy(y_true, y_pred):
    return sum(int(a==b) for a,b in zip(y_true,y_pred))/max(1,len(y_true))
