import pickle
import numpy as np
import pandas as pd
import torch
from ravenpackapi import RPApi

# we temporarily cannot share the api key
api = RPApi(api_key='')

def read_data(args):
    path = args.path + args.region + '/' + args.sector + '/'
    batch = args.batch_size
    with open(path + "input_ef.pkl", "rb") as f:
        input_ef = np.array(pickle.load(f))
  
    if args.mode == 'price_spike':
        with open(path + "input_y_price.pkl", "rb") as f:
            label_y = np.array(pickle.load(f))
    elif args.mode == 'volume_spike':
         with open(path + "input_y_volume.pkl", "rb") as f:
            label_y = np.array(pickle.load(f))
    else:
        raise ValueError
        
    
    with open(path + "companies_list.pkl", 'rb') as f:
        company_list = pickle.load(f)
    with open(path + "input_et.pkl", "rb") as f:
        input_et = np.array(pickle.load(f))
    with open(path + "input_pt.pkl", "rb") as f:
        input_pt = np.array(pickle.load(f))
    with open(path + "input_co.pkl", 'rb') as f:
        input_co = np.array(pickle.load(f))
    with open(path + "input_vt.pkl", "rb") as f:
        input_vt = np.array(pickle.load(f))
    with open(path + "companies_tensor.pkl", 'rb') as f:
        companies_idx = pickle.load(f)
    with open(path + "entity_to_score_with_market_price_two_labels_ALL_COMPS_MP.pkl", "rb") as f:
        ent2score = pickle.load(f)
    with open(path + "input_return.pkl", "rb") as f:
        ret = pickle.load(f)

    company_dates = list(sorted(ent2score.keys(), key=lambda x: x[1]))

    Input_ef = []
    Input_et = []
    Input_co = []
    Input_pt = []
    Input_vt = []
    Input_comp_idx = []
    Label_y = []
    for idx in range(0, len(input_ef), batch):
        Input_ef.append(input_ef[idx: idx + batch])
        Input_et.append(input_et[idx: idx + batch])
        Input_co.append(input_co[idx: idx + batch])
        Input_pt.append(input_pt[idx: idx + batch])
        Input_vt.append(input_vt[idx: idx + batch])
        Input_comp_idx.append(companies_idx[idx: idx + batch])
        Label_y.append(label_y[idx: idx + batch])


    return Input_ef, Input_et, Input_co, Input_pt, Input_vt, Label_y, company_list, company_dates, Input_comp_idx, ret


def get_sharpe_ratio(ret_dict, dates):
    dates = sorted(dates)
    res = []
    profit = []
    money = 1.0
    now = 0
    for key, value in sorted(ret_dict.items()):
        idx = dates.index(key)
        for _ in range(now, idx):
            profit.append(money)
        now = idx
        value = sum(value) / len(value)
        money = money * (1.0 + value)
        res.append(money)

    for _ in range(now, len(dates)):
        profit.append(money)

    assert len(profit) == len(dates)

    res = pd.DataFrame(np.array(res))
    r = res.diff()
    sr = r.mean() / r.std()
    return sr, profit

def make_trading_decision(date, dates, rets, ret_data, ef, idx):

    ef = ef.view(-1, 7)
    css = ef.cpu().numpy().tolist()

    if css[-1][3] >= 0:
        ret = ret_data[idx]
    else:
        ret = -1 * ret_data[idx]
    
    rets[date].append(ret)

    if date not in dates:
        dates.append(date)

    return dates, rets

def compute_factor_weights(attn, factors, comp_name_dict, company_list, num_x):

    #################### POSITIVE ########################
    attn = attn.cpu().squeeze(0)
    # print(attn.size())
    v, i = torch.topk(attn.flatten(), 10)
    loc = np.array(np.unravel_index(i.numpy(), attn.shape)).T

    universe = []
    for value in loc:
        tp, rv, cn = value
        cn = company_list[cn]
        universe.append(cn)
    mapping = api.get_entity_mapping(universe)

    v_num = 0
    for value in loc:
        tp, rv, cn = value
        factors[(tp, rv, cn)] = (factors[(tp, rv, cn)] * num_x + v[v_num]) / (num_x + 1)
        comp_name_dict[cn] = mapping.matched[v_num].name
        print('Time Period: {}, Relational View: {}, Company Name: {}'.format(tp, rv, mapping.matched[v_num].name))
        v_num += 1
    #################### NEGATIVE ########################
    v, i = torch.topk(attn.flatten(), 10, largest=False)
    loc = np.array(np.unravel_index(i.numpy(), attn.shape)).T

    universe = []
    for value in loc:
        tp, rv, cn = value
        cn = company_list[cn]
        universe.append(cn)
    mapping = api.get_entity_mapping(universe)

    v_num = 0
    for value in loc:
        tp, rv, cn = value
        factors[(tp, rv, cn)] = (factors[(tp, rv, cn)] * num_x + v[v_num]) / (num_x + 1)
        comp_name_dict[cn] = mapping.matched[v_num].name
        print('Time Period: {}, Relational View: {}, Company Name: {}'.format(tp, rv, mapping.matched[v_num].name))
        v_num += 1

    return factors, comp_name_dict
