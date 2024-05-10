from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

faces = fetch_lfw_people(min_faces_per_person=100)

X = faces.data
y = faces.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We will use RandomizedPCA to consider only most important features of the dataset. 

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')

# Let’s take advantage of pipelines that combines multiple steps together and helps to write a clear code 
model = make_pipeline(pca, svc)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Let’s print out the Accuracy Score of our model
print( "Accuracy Score: " + str(accuracy_score(predictions, y_test)) )

# Let’s print out the values of Predicted vs Actual 
print( "Predictions\n-------------------------")
print("Predicted,Actual")

for cnt in range(len(predictions)):	
    if predictions[cnt] == y_test[cnt]:
        predicted = faces.target_names[predictions[predictions[cnt]]]
        actual = faces.target_names[y_test[y_test[cnt]]]
        print("%s,%s" % (str(predicted).strip(),str(actual).strip() )  )