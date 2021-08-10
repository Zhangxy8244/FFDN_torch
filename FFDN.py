import torch
import torch.nn as nn

from layers.FuzzyLayer import FuzzyLayer


class FusedFuzzyDeepNet(nn.Module):
    def __init__(self, input_vector_size, fuzz_vector_size, num_class, fuzzy_layer_input_dim=1,
                 fuzzy_layer_output_dim=1,
                 dropout_rate=0.5):

        super(FusedFuzzyDeepNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_vector_size = input_vector_size
        self.fuzz_vector_size = fuzz_vector_size
        self.num_class = num_class
        self.fuzzy_layer_input_dim = fuzzy_layer_input_dim
        self.fuzzy_layer_output_dim = fuzzy_layer_output_dim

        self.dropout_rate = dropout_rate

        self.fuzz_init_linear_layer = nn.Linear(self.input_vector_size, self.fuzz_vector_size)

        fuzzy_rule_layers = []
        for i in range(self.fuzz_vector_size):
            fuzzy_rule_layers.append(FuzzyLayer(fuzzy_layer_input_dim, fuzzy_layer_output_dim))
        self.fuzzy_rule_layers = nn.ModuleList(fuzzy_rule_layers)

        self.dl_linear_1 = nn.Linear(self.input_vector_size, self.input_vector_size)
        self.dl_linear_2 = nn.Linear(self.input_vector_size, self.input_vector_size)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.fusion_layer = nn.Linear(self.input_vector_size * 2, self.input_vector_size)
        self.output_layer = nn.Linear(self.input_vector_size, self.num_class)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):

        fuzz_input = self.fuzz_init_linear_layer(input)
        fuzz_output = torch.zeros(input.size(), dtype=torch.float, device=self.device)
        for col_idx in range(fuzz_input.size()[1]):
            col_vector = fuzz_input[:, col_idx:col_idx + 1]
            fuzz_col_vector = self.fuzzy_rule_layers[col_idx](col_vector).unsqueeze(0).view(-1, 1)
            fuzz_output[:, col_idx:col_idx + 1] = fuzz_col_vector

        dl_layer_1_output = torch.sigmoid(self.dl_linear_1(input))
        dl_layer_2_output = torch.sigmoid(self.dl_linear_2(dl_layer_1_output))
        dl_layer_2_output = self.dropout_layer(dl_layer_2_output)

        cat_fuzz_dl_output = torch.cat([fuzz_output, dl_layer_2_output], dim=1)

        fused_output = torch.sigmoid(self.fusion_layer(cat_fuzz_dl_output))
        fused_output = torch.relu(fused_output)

        output = self.log_softmax(self.output_layer(fused_output))

        return output
