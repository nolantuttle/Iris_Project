import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

class TestIrisPipeline(unittest.TestCase):
    def setUp(self):
        self.iris = load_iris()
        self.x = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names
    
    def test_data_loading(self):
        self.assertEqual(self.x.shape, (150, 4))    # 150 samples, 4 features in the set
        self.assertEqual(self.y.shape, (150,))
        self.assertEqual(len(self.feature_names), 4)
        self.assertEqual(len(self.target_names), 3)

    def test_missing_and_duplicates(self):
        data = pd.DataFrame(self.x, columns=self.feature_names)
        self.assertTrue(data.isnull().sum().sum() == 0)

        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            data.drop_duplicates(inplace=True)

        self.assertEqual(data.duplicated().sum(), 0)

    def test_data_splittig(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.2, random_state = 42)
        self.assertEqual(len(x_train), 120) # 80% of 150
        self.assertEqual(len(x_test), 30)   # 20% of 150

    def test_scaling(self):
        x_train, x_test, _, _ = train_test_split(self.x, self.y, test_size = 0.2, random_state = 42)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        self.assertAlmostEqual(x_train_scaled.mean(), 0, delta=0.1)
        self.assertAlmostEqual(x_train_scaled.std(), 1, delta=0.1)

    def test_model_training_and_evaluation(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.2, random_state = 42)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = LogisticRegression()
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)

        # Test Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.9)

        cm = confusion_matrix(y_test, y_pred)
        self.assertEqual(cm.shape, (3,3))

if __name__ == '__main__':
    unittest.main()

