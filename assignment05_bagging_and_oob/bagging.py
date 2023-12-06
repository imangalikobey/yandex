import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Shuffle indices to create a random split
            indices = np.random.choice(data_length, data_length, replace=True)
            self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag_indices in self.indices_list:
            model = model_constructor()
            data_bag, target_bag = data[bag_indices], target[bag_indices]
            self.models_list.append(model.fit(data_bag, target_bag))  # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = np.array([model.predict(data) for model in self.models_list])
        return np.mean(predictions, axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates a list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during the training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for i in range(len(self.data)):
            for j in range(self.num_bags):
                if i not in self.indices_list[j]:
                    list_of_predictions_lists[i].append(self.models_list[j].predict([self.data[i]]))
                    
        self.list_of_predictions_lists = list_of_predictions_lists
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute the average prediction for every object from the training set.
        If the object has been used in all bags during the training phase, return None instead of the prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([np.mean(predictions) if predictions else None 
                                         for predictions in self.list_of_predictions_lists])
        
    def OOB_score(self):
        '''
        Compute the mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        valid_indices = [i for i, predictions in enumerate(self.list_of_predictions_lists) if predictions]
        squared_errors = np.square(self.target[valid_indices] - self.oob_predictions[valid_indices])
        return np.mean(squared_errors)

# Example usage:
# bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
# bagging_regressor.fit(LinearRegression, X, y)
# oob_score = bagging_regressor.OOB_score()
# predictions = bagging_regressor.predict(new_data)
