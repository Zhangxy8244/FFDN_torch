import torch
import torch.nn as nn


class FuzzyLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FuzzyLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        fuzzy_degree_weights = torch.Tensor(self.input_dim, self.output_dim)
        self.fuzzy_degree = nn.Parameter(fuzzy_degree_weights)
        sigma_weights = torch.Tensor(self.input_dim, self.output_dim)
        self.sigma = nn.Parameter(sigma_weights)

        # initialize fuzzy degree and sigma parameters
        nn.init.xavier_uniform_(self.fuzzy_degree)  # fuzzy degree init
        nn.init.ones_(self.sigma)  # sigma init

    def forward(self, input):
        fuzzy_out = []
        for variable in input:
            fuzzy_out_i = torch.exp(-torch.sum(torch.sqrt((variable - self.fuzzy_degree) / (self.sigma ** 2))))
            if torch.isnan(fuzzy_out_i):
                fuzzy_out.append(variable)
            else:
                fuzzy_out.append(fuzzy_out_i)
        return torch.tensor(fuzzy_out, dtype=torch.float)
