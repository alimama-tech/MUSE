import numpy as np
import torch

from model.base_model.dice import activation_layer

class FCRepeat(torch.nn.Module):
    def __init__(self, dim_in, shapes: list, acts=None, weights=None) -> None:
        super().__init__()
        self.linears = torch.nn.ModuleList()
        self.acts = acts
        self.act_layers = torch.nn.ModuleList()
        self.weights = weights
        self.dims = [dim_in]
        self.dims.extend(shapes)
        self.name = ""

        for index, (shape_in, shape_out) in enumerate(
            zip(self.dims[:-1], self.dims[1:])
        ):
            self.linears.append(torch.nn.Linear(shape_in, int(shape_out)))
            if self.acts is None or self.acts[index] is None or self.acts[index] == "":
                self.act_layers.append(torch.nn.Identity())
            else:
                self.act_layers.append(activation_layer(self.acts[index], shape_out, 1000, True))

    def reset_parameters(self):
        for index, (shape_in, shape_out) in enumerate(
            zip(self.dims[:-1], self.dims[1:])
        ):
            weight = np.random.randn(int(shape_in), int(shape_out)).astype(
                np.float32
            ) / np.sqrt(shape_in)
            self.linears[index].weight.data.copy_(torch.tensor(weight).T)
            torch.nn.init.constant_(self.linears[index].bias, 0.1)
            if self.acts is not None and self.acts[index] is not None:
                if self.acts[index] in ["prelu", "dice"]:
                    self.act_layers[index].reset_parameters()

    def forward(self, data):
        dout = data
        for layer_index in range(len(self.linears)):
            dout = self.linears[layer_index](dout) 
            dout = self.act_layers[layer_index](dout) 
        return dout

def fc_repeats(feature_in, shape: list, acts=None, weights=None) -> FCRepeat:
    return FCRepeat(feature_in, shapes=shape, acts=acts, weights=weights)

class MultiHeadAtt(torch.nn.Module):
    def __init__(
        self,
        query_in_shape,
        fact_in_shape,
        fc_query_shapes: list,
        fc_fact_shapes: list,
        side_in_shape=None,
        side_shapes=None,
        beta=0,
        use_cosine=False,
        use_mm_cosine=False,
        scale=1,
    ) -> None:
        super().__init__()
        self.query_in_shape = query_in_shape
        self.fact_in_shape = fact_in_shape
        self.fc_query_shapes = fc_query_shapes
        self.fc_fact_shapes = fc_fact_shapes
        self.side_shapes = side_shapes
        self.beta = beta
        self.use_cosine = use_cosine
        self.use_mm_cosine = use_mm_cosine
        self.scale = scale
        self.name = ""
        self.head_num = int(max(self.query_in_shape * scale / fc_query_shapes[0], 1))

        self.query_fcs = torch.nn.ModuleList()
        self.fact_fcs = torch.nn.ModuleList()
        if self.side_shapes is not None:
            self.side_info_fcs = torch.nn.ModuleList()
        if self.use_cosine:
            self.tau = torch.nn.Parameter(torch.empty((1,)))
        for k in range(self.head_num):
            query_layer = FCRepeat(query_in_shape, self.fc_query_shapes)
            fact_layer = FCRepeat(fact_in_shape, self.fc_fact_shapes)
            self.query_fcs.append(query_layer)
            self.fact_fcs.append(fact_layer)
            if self.side_shapes is not None:
                self.side_info_fcs.append(FCRepeat(side_in_shape, side_shapes))
        if self.use_mm_cosine:
            self.mm_tau1 = torch.nn.Parameter(torch.empty([1]))
            self.mm_tau2 = torch.nn.Parameter(torch.empty([1]))
            self.mm_tau3 = torch.nn.Parameter(torch.empty([1]))

    def reset_parameters(self):
        for index in range(self.head_num):
            self.query_fcs[index].reset_parameters()
            self.fact_fcs[index].reset_parameters()
            if self.side_shapes is not None:
                self.side_info_fcs[index].reset_parameters()
        if self.use_cosine:
            torch.nn.init.constant_(self.tau, 0.1)
        if self.use_mm_cosine:
            torch.nn.init.constant_(self.mm_tau1, 0.1)
            torch.nn.init.constant_(self.mm_tau2, 0.1)
            torch.nn.init.constant_(self.mm_tau3, 0.0)

    def forward(self, query, fact, side_info=None, indicator=None, mm_cosine=None):
        assert any(
            [
                all([self.use_mm_cosine == False, mm_cosine is None]),
                all([self.use_mm_cosine == True, mm_cosine is not None]),
            ]
        )
        fc_queries = []
        for index, module in enumerate(self.query_fcs):
            fc_queries.append(module(query))
        fc_facts = []
        for module in self.fact_fcs:
            fc_fact = module(fact)
            if indicator is not None:
                fc_fact = torch.index_select(fc_fact, 0, indicator)
            fc_facts.append(fc_fact)
        if side_info is not None:
            fc_sides = []
            for module in self.side_info_fcs:
                fc_side = module(side_info)
                if indicator is not None:
                    fc_side = torch.index_select(fc_side, 0, indicator)
                fc_sides.append(fc_side)
        outs = []
        for k in range(self.head_num):
            if side_info is not None:
                dot1 = torch.matmul(fc_facts[k], torch.transpose(fc_queries[k], -1, -2))
                dot2 = torch.matmul(fc_sides[k], torch.transpose(fc_queries[k], -1, -2))
                alphas1 = torch.nn.functional.softmax(dot1, dim=1) + 0.0000001
                alphas2 = torch.nn.functional.softmax(dot2, dim=1) + 0.0000001
                alphas = self.beta * alphas1 + (1.0 - self.beta) * alphas2
                out = torch.matmul(torch.transpose(fc_facts[k], -1, -2), alphas)
            elif self.use_cosine:
                a = torch.nn.functional.normalize(fc_facts[k], dim=2)
                b = torch.nn.functional.normalize(fc_queries[k], dim=2)
                dot = torch.sum(a * b, dim=2, keepdim=True) / self.tau
                alphas = torch.nn.functional.softmax(dot, dim=1) + 0.0000001
                out = torch.matmul(torch.transpose(fc_facts[k], -1, -2), alphas)
            else:
                dot = torch.matmul(fc_facts[k], fc_queries[k].transpose(-1, -2))
                if mm_cosine is not None:
                    assert self.use_mm_cosine == True
                    mm_tau3 = torch.nn.functional.sigmoid(self.mm_tau3)
                    mm_cosine = torch.unsqueeze(
                        mm_cosine[0] / self.mm_tau1 + mm_cosine[1] / self.mm_tau2,
                        dim=-1,
                    )
                    alphas = (
                        torch.nn.functional.softmax(
                            mm_tau3 * dot + (1 - mm_tau3) * mm_cosine, dim=1
                        )
                        + 0.0000001
                    )
                else:
                    alphas = torch.nn.functional.softmax(dot, dim=1) + 0.0000001
                out = torch.matmul(fc_facts[k].transpose(-1, -2), alphas)
            outs.append(out)
        return torch.squeeze(torch.concat(outs, dim=1), dim=2)

def multi_head_att(
    query_in_shape,
    fact_in_shape,
    fc_query_shapes: list,
    fc_fact_shapes: list,
    side_in_shape=None,
    side_shapes=None,
    beta=0,
    use_cosine=False,
    use_mm_cosine=False,
    scale=1,
):
    return MultiHeadAtt(
        query_in_shape=query_in_shape,
        fact_in_shape=fact_in_shape,
        fc_query_shapes=fc_query_shapes,
        fc_fact_shapes=fc_fact_shapes,
        side_in_shape=side_in_shape,
        side_shapes=side_shapes,
        beta=beta,
        use_cosine=use_cosine,
        use_mm_cosine=use_mm_cosine,
        scale=scale,
    )

class MultiHeadAttV2(torch.nn.Module):
    def __init__(
        self, quer_in_features, fact_in_features, fc_query_shapes, fc_fact_shapes, attn_score_cross=False
    ):
        super().__init__()
        self.attn_score_cross = attn_score_cross
        self.heads = int(max(quer_in_features // fc_query_shapes[0], 1))
        self.query_layers = torch.nn.ModuleList()
        self.fact_layers = torch.nn.ModuleList()
        self.value_layers = torch.nn.ModuleList()
        for _ in range(self.heads):
            self.query_layers.append(fc_repeats(quer_in_features, fc_query_shapes))
            self.fact_layers.append(fc_repeats(fact_in_features, fc_fact_shapes))
            self.value_layers.append(fc_repeats(fact_in_features, fc_fact_shapes))
        if attn_score_cross:
            self.cosine_tau1 = torch.nn.Parameter(torch.empty([2, self.heads]))
            self.cosine_tau2 = torch.nn.Parameter(torch.empty([3, self.heads]))

    def reset_parameters(self):
        for module in self.query_layers:
            module.reset_parameters()
        for module in self.fact_layers:
            module.reset_parameters()
        for module in self.value_layers:
            module.reset_parameters()
        if self.attn_score_cross:
            torch.nn.init.constant_(self.cosine_tau1, 0.1)
            torch.nn.init.constant_(self.cosine_tau2, 1.0)

    def forward(self, query: torch.Tensor, fact: torch.Tensor, mask: torch.Tensor = None, mm_cosine: torch.Tensor =None):
        outs = []
        for k in range(self.heads):
            fc_query = self.query_layers[k](query)
            fc_query = fc_query * torch.nn.functional.sigmoid(fc_query)
            fc_fact = self.fact_layers[k](fact)
            fc_fact = fc_fact * torch.nn.functional.sigmoid(fc_fact)
            fc_value = self.value_layers[k](fact)
            fc_value = fc_value * torch.nn.functional.sigmoid(fc_value)
            dot = torch.matmul(fc_fact, fc_query.transpose(-1, -2))
            if mask is not None:
                dot += - 1e9 * (1-mask.unsqueeze(-1).float())
            if self.attn_score_cross and mm_cosine is not None:
                if len(mm_cosine) > 1:
                    a_1 = self.cosine_tau1[0, k].view(1, -1)
                    a_2 = self.cosine_tau1[1, k].view(1, -1)
                    mm_attn_bias = mm_cosine[0] / a_1 + mm_cosine[1] / a_2
                else:
                    a_1 = self.cosine_tau1[0, k].view(1, -1)
                    mm_attn_bias = mm_cosine[0] / a_1
                
                mm_attn_bias = mm_attn_bias.view(-1, dot.shape[1], 1)
            
                b_1 = self.cosine_tau2[0, k].view(1, 1, 1)
                b_2 = self.cosine_tau2[1, k].view(1, 1, 1)                
                b_3 = self.cosine_tau2[2, k].view(1, 1, 1)
                dot = b_1*dot + b_2*mm_attn_bias + b_3*dot*mm_attn_bias
                # print('cosine_tau', self.cosine_tau1, self.cosine_tau2)
                
            alphas = torch.nn.functional.softmax(dot, dim=1) + 0.0000001
            out = torch.matmul(fc_value.transpose(-1, -2), alphas)
            outs.append(out)
        return torch.squeeze(torch.concat(outs, dim=1), dim=2)
    
    def calc_attn_score(self, query: torch.Tensor, fact: torch.Tensor, mask: torch.Tensor = None):
        alphas_list = []
        for k in range(self.heads):
            fc_query = self.query_layers[k](query)
            fc_query = fc_query * torch.nn.functional.sigmoid(fc_query)
            fc_fact = self.fact_layers[k](fact)
            fc_fact = fc_fact * torch.nn.functional.sigmoid(fc_fact)
            dot = torch.matmul(fc_fact, fc_query.transpose(-1, -2))
            if mask is not None:
                dot += - 1e9 * (1-mask.unsqueeze(-1).float())
            alphas = torch.nn.functional.softmax(dot, dim=1) + 0.0000001
            alphas_list.append(alphas.squeeze(dim=-1))
        alpha_avg = torch.stack(alphas_list).sum(dim=0) / self.heads
        return alpha_avg

def multi_head_att_v2(
    quer_in_features, fact_in_features, fc_query_shapes, fc_fact_shapes, attn_score_cross=False
):
    return MultiHeadAttV2(
        quer_in_features, fact_in_features, fc_query_shapes, fc_fact_shapes, attn_score_cross
    )