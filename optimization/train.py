import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from model import WeightedLogisticRegression
import matplotlib.pyplot as plt

NUM_FEATURES = 13
NUM_LABELS = 1
DEEGRES_FREEDOM = 2
ONE = torch.tensor([1.])
#Avoid using global seeds
torch.manual_seed(1)

# TODO: Santiago, refactor this method.
def assign_weights(row, label, w):
    """Weight assignation
    
    Depending on the aprticular value of a row
    It is going to have a different weight assigned.
    """
    if label == 0:
        if row[0] == 1:
            if row[7] == 1:
                return 1/(0.25 - w[0])
            else:
                return 1/w[0]
        else:
            return ONE

    if label == 1:
        if row[1] == 0:
            return ONE 
        else:
            if row[7] == 1:
                return 1/(0.25 - w[1])
            else:
                return 1/w[1]


def train_model(model, data, labels, mode):
    """training procedure to produce uncertanty quantification.

    In order to produce uncertainty quantification on the coefficients,
    It is necessary to alternate the differiantiation between the coe
    fficients and the debiasing weights.

    Args:
        model (_type_): _description_
    """
    # BCE loss for fitting the logistic regression
    loss_function = torch.nn.BCELoss()

    # MSE loss for the absolute value of the coefficients.
    second_loss_function = torch.nn.MSELoss()

    # Two optimizers, they have the relevant parameters to optimize.
    optimizer = optim.SGD(params=model.linear.parameters(), lr=0.001)
    optimizer_2 = torch.optim.Adam(params=[model.weights], lr=0.001)

    for i in range(20):
        
        
        for epoch in range(10):
            for instance, label in zip(data, labels):
                # Step 1. Remember that PyTorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2.
                features = torch.tensor(instance.astype(np.double)).float()
                # Step 3. Run our forward pass.
                if i == 0:
                    weight = 1.
                else:
                    weight = assign_weights(features, label, model.weights)
                probs = model(features) * weight

                with torch.no_grad():            
                        probs.clamp_(0, 1)
               
                target = torch.tensor(label, dtype = torch.long).expand((1)).float()

                # Step 4. Compute the loss, gradients, and update the parameters by
                loss = loss_function(probs, target)

                if loss.requires_grad:
                    loss.backward(create_graph=True)
                    optimizer.step()

        # Put an assert ensuring the betas have 0 gradients.
        coeff = next(model.linear.parameters())
        # import pdb; pdb.set_trace()
        loss_w = second_loss_function(-coeff[0][7], torch.tensor(0.))
        print(loss_w)
        loss_w.backward()
        optimizer_2.step()

        with torch.no_grad():            
            # are theese good assumptions?
            model.weights.clamp_(coeff[0][7], 0.25 - coeff[0][7])

    return coeff[0][0]
    
if __name__ == '__main__':
    """The following file executes a training pipe-line for a debiased logistic regresssion.
    The process consists in alternate the differentiattion procedure between the parameters
    of the logistic regression and the debiasing weights.
    The idea is that the way the weights are assigned produce restrictions that are being
    solved by the automatic Pytorch differentiation.
    """
    data = pd.read_csv("german_credit_data.csv", index_col=0)
    data_labels = pd.read_csv("german_credit.csv")
    data_dummies = pd.get_dummies(data["Sex"])
    data_dummies[["free", "own", "rent"]] = pd.get_dummies(data["Housing"])
    data_dummies[["little", "moderate", "quite rich", "rich"]] = pd.get_dummies(data["Saving accounts"])
    data_dummies[["Age Bucket A", "Age Bucket B", "Age Bucket c", "Age Bucket D"]] = pd.get_dummies(pd.cut(data.Age, [18, 25, 45, 60, 75]))
    data_dummies["Creditability"] = data_labels["Creditability"]

    females = data_dummies[data_dummies.female == 1]
    females_good = females[females.Creditability == 1]
    females_bad = females[females.Creditability == 0]
    males = data_dummies[data_dummies.female == 0]
    males_good = males[males.Creditability == 1]
    males_bad = males[males.Creditability == 0]
    ma_good_sample = males_good.sample(frac = 0.25)
    fem_bad_sample = females_bad.sample(frac = 0.25)
    sex_skewed_data = pd.concat([ma_good_sample, males_bad, females_good, fem_bad_sample])
    print("Data feat:", sex_skewed_data.columns)
    var = data_dummies.columns
    VAR = var[:-1]

    data = sex_skewed_data[VAR].to_numpy()
    labels = sex_skewed_data["Creditability"]

    model = WeightedLogisticRegression(NUM_FEATURES, NUM_LABELS, DEEGRES_FREEDOM)

    coef = train_model(model, data, labels, "min")
    print("Final", coef)




